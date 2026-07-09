"""
brain/memory/store.py  — v3

v3 changes (reply-system-v6):
  retrieve_hybrid()    — 70 % semantic + 30 % recency weighted RAG.
                         Blends cosine-similarity results with a time-decay
                         boost so recent messages from the same conversation
                         surface even when they are a weak semantic match.
  retrieve_thread()    — fetch 2 messages before and 2 after a given match.
                         Returns mini-thread instead of an isolated message.
                         Uses ChromaDB timestamp-range metadata filtering.
  _retrieve_recent()   — uncapped retrieval of the last N messages within
                         a chat/sender scope, scored by recency alone.

v2 changes (from original):
  _make_id()     — now includes chat_name in hash.
                   Prevents collisions when same sender sends same content
                   at the same second in two different chats.

  add_messages() — deduplicates IDs within each batch before upsert.
                   ChromaDB 1.5.x raises ValueError (not just a warning)
                   when a single upsert() call contains duplicate IDs.
                   Without this fix, entire chat files were silently skipped.

Action after patching:
    python -m cli.main store data/exports/ --you "Rishit Babbar" --clear
    # --clear wipes old records (they had v1 ID format) and rebuilds cleanly.
"""
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from brain.profiler.models import MessageRecord


EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "persona_messages"
DEFAULT_TOP_K   = 5
BATCH_SIZE      = 128


# ── Main class ─────────────────────────────────────────────────────────────────

class PersonaStore:
    """
    ChromaDB-backed vector store for PersonaOS message memory.

    Typical usage:
        store = PersonaStore("data/db/chroma")
        store.add_messages(records, sender_filter="Rishit Babbar")
        results = store.retrieve_similar("what are you up to tonight", top_k=5)
        for r in results:
            print(f"[{r['similarity']:.0%}] {r['content']}")
    """

    def __init__(
        self,
        chroma_path: str | Path = "data/db/chroma",
        model_name: str = EMBEDDING_MODEL,
    ):
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(path=str(self.chroma_path))

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_messages(
        self,
        records: list[MessageRecord],
        sender_filter: str | None = None,
    ) -> int:
        """
        Embed and store MessageRecords in ChromaDB.

        v2 change: deduplicates IDs within each batch before calling upsert().
        ChromaDB 1.5.x raises ValueError for duplicate IDs within a single
        upsert() call. Deduplication here keeps the first occurrence of any
        duplicate and silently drops the rest within the same batch.
        Cross-batch duplicates are handled correctly by ChromaDB's upsert
        semantics (later write wins).

        Args:
            records:       Parsed MessageRecord list (multi-sender OK)
            sender_filter: Only embed messages from this sender.

        Returns:
            Number of messages stored in this call.
        """
        to_embed = [
            r for r in records
            if r.content.strip()
            and len(r.content.strip()) >= 3
            and not r.has_media
            and (sender_filter is None or r.sender == sender_filter)
        ]

        if not to_embed:
            return 0

        total_stored = 0

        for i in range(0, len(to_embed), BATCH_SIZE):
            batch = to_embed[i : i + BATCH_SIZE]

            # ── Deduplicate IDs within this batch ─────────────────────────────
            # ChromaDB 1.5.x throws ValueError if the same ID appears more than
            # once in a single upsert() call. We keep first occurrence only.
            seen: set[str] = set()
            deduped: list[MessageRecord] = []
            for r in batch:
                rid = _make_id(r)
                if rid not in seen:
                    seen.add(rid)
                    deduped.append(r)
            batch = deduped

            if not batch:
                continue

            ids        = [_make_id(r) for r in batch]
            documents  = [r.content for r in batch]
            metadatas  = [_make_metadata(r) for r in batch]
            embeddings = self.model.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            total_stored += len(batch)

        return total_stored

    # ── Read ───────────────────────────────────────────────────────────────────

    def retrieve_similar(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
        sender_filter: str | None = None,
        chat_filter: str | None = None,
    ) -> list[dict]:
        """
        Find the most semantically similar past messages to `text`.

        Args:
            text:          Incoming message or any query text
            top_k:         Number of results to return (default 5)
            sender_filter: Only retrieve messages from this sender
            chat_filter:   Only retrieve messages from this chat/contact

        Returns:
            List of dicts sorted by similarity (most similar first).
        """
        if self.collection.count() == 0:
            return []

        if not text or not text.strip():
            return []

        where = _build_where(sender_filter, chat_filter)
        query_embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        safe_top_k = min(top_k, self.collection.count())

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=safe_top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        output = [
            {
                "content":    doc,
                "sender":     meta.get("sender", ""),
                "chat_name":  meta.get("chat_name", ""),
                "timestamp":  meta.get("timestamp", ""),
                "platform":   meta.get("platform", ""),
                "distance":   round(dist, 4),
                "similarity": round(max(0.0, 1 - dist), 4),
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]

        return sorted(output, key=lambda x: x["similarity"], reverse=True)

    # ── Hybrid retrieval (v3) ──────────────────────────────────────────────────

    def retrieve_hybrid(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
        sender_filter: str | None = None,
        chat_filter: str | None = None,
        semantic_weight: float = 0.70,
        recency_days: int = 30,
    ) -> list[dict]:
        """
        Blend semantic similarity results with recency-weighted results.

        70 % semantic (cosine similarity via embeddings)
        30 % recency  (time-decay boost: 1.0 → 0.0 over `recency_days` days)

        Deduplicates by content[:80] so the same message doesn't appear twice.

        Args:
            text:            Incoming message / query text
            top_k:           Number of results to return
            sender_filter:   Only this sender's messages
            chat_filter:     Only this chat/contact
            semantic_weight: Weight of semantic scores (default 0.70)
            recency_days:    Days over which recency boost decays (default 30)

        Returns:
            List of dicts with blended scores, sorted highest first.
        """
        if self.collection.count() == 0:
            return []

        if not text or not text.strip():
            return []

        # ── Layer 1: semantic retrieval ────────────────────────────────────────
        semantic_results = self.retrieve_similar(
            text=text,
            top_k=max(top_k * 2, 10),   # over-fetch for blending headroom
            sender_filter=sender_filter,
            chat_filter=chat_filter,
        )

        # ── Layer 2: recency retrieval ────────────────────────────────────────
        recency_results = self._retrieve_recent(
            sender_filter=sender_filter,
            chat_filter=chat_filter,
            limit=top_k * 2,
            days=recency_days,
        )

        # ── Blend scores ──────────────────────────────────────────────────────
        now = datetime.now(timezone.utc)
        blended: dict[str, dict] = {}          # key = content[:80] for dedup

        for r in semantic_results:
            key = r["content"][:80]
            blended[key] = {
                **r,
                "semantic_score": r["similarity"],
                "recency_score": 0.0,
                "_blended": r["similarity"],
            }

        for r in recency_results:
            key = r["content"][:80]
            # Compute recency boost: 1.0 for today → 0.0 at recency_days ago
            try:
                ts = datetime.fromisoformat(r.get("timestamp", ""))
                days_ago = max(0.0, (now - ts).total_seconds() / 86400)
                recency_score = max(0.0, 1.0 - (days_ago / recency_days))
            except (ValueError, TypeError):
                recency_score = 0.0

            if key in blended:
                existing = blended[key]
                existing["recency_score"] = recency_score
                existing["_blended"] = (
                    semantic_weight * existing["semantic_score"]
                    + (1 - semantic_weight) * recency_score
                )
            else:
                blended[key] = {
                    **r,
                    "semantic_score": 0.0,
                    "recency_score": recency_score,
                    "_blended": (1 - semantic_weight) * recency_score,
                }

        # ── Sort by blended score, return top_k ───────────────────────────────
        ranked = sorted(blended.values(), key=lambda x: -x["_blended"])
        for r in ranked:
            r["similarity"] = round(min(r["_blended"], 1.0), 4)
            del r["_blended"]

        return ranked[:top_k]

    def _retrieve_recent(
        self,
        sender_filter: str | None = None,
        chat_filter: str | None = None,
        limit: int = 10,
        days: int = 30,
    ) -> list[dict]:
        """
        Fetch the most recent messages matching the given filters.

        Uses ChromaDB's get() to pull all matching metadata, sorts in
        Python by timestamp, and returns the most recent `limit` messages.
        No embedding query — pure recency ordering.
        """
        if self.collection.count() == 0:
            return []

        where = _build_where(sender_filter, chat_filter)

        try:
            if where:
                result = self.collection.get(
                    where=where,
                    include=["documents", "metadatas"],
                )
            else:
                result = self.collection.get(
                    include=["documents", "metadatas"],
                )
        except Exception:
            return []

        if not result or not result["documents"]:
            return []

        docs  = result["documents"]
        metas = result["metadatas"]

        # Sort by timestamp descending, take most recent
        paired = list(zip(docs, metas))
        paired.sort(
            key=lambda p: p[1].get("timestamp", ""),
            reverse=True,
        )

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        output: list[dict] = []
        for doc, meta in paired:
            try:
                ts = datetime.fromisoformat(meta.get("timestamp", ""))
                if ts < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
            output.append({
                "content":    doc,
                "sender":     meta.get("sender", ""),
                "chat_name":  meta.get("chat_name", ""),
                "timestamp":  meta.get("timestamp", ""),
                "platform":   meta.get("platform", ""),
                "distance":   0.0,
                "similarity": 0.0,   # recency-only, no semantic score
            })
            if len(output) >= limit:
                break

        return output

    # ── Thread retrieval (v3) ─────────────────────────────────────────────────

    def retrieve_thread(
        self,
        anchor: dict,
        window: int = 2,
    ) -> list[dict]:
        """
        Fetch messages surrounding a given anchor message in the same chat.

        Given a message dict (as returned by retrieve_similar / retrieve_hybrid),
        fetch `window` messages before and `window` messages after it in the
        same chat, ordered chronologically. The anchor itself is not included
        (the caller already has it).

        Returns up to (window * 2) messages. Fewer if the chat is short or
        the anchor is near the beginning/end.

        Args:
            anchor: A dict with 'chat_name' + 'timestamp' keys.
            window: Messages to fetch on each side (default 2 → 4 total).

        Returns:
            List of dicts in chronological order.
        """
        if self.collection.count() == 0:
            return []

        chat_name  = anchor.get("chat_name", "")
        ts_str     = anchor.get("timestamp", "")

        if not chat_name or not ts_str:
            return []

        try:
            anchor_ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return []

        # ── Fetch messages before anchor ──────────────────────────────────────
        before = self._retrieve_by_time_range(
            chat_name=chat_name,
            before_ts=anchor_ts,
            after_ts=None,
            limit=window,
        )

        # ── Fetch messages after anchor ───────────────────────────────────────
        after = self._retrieve_by_time_range(
            chat_name=chat_name,
            before_ts=None,
            after_ts=anchor_ts,
            limit=window,
        )

        # Already in chronological order within each batch.
        # before is newest-first (reverse chron), so reverse it.
        before.reverse()
        return before + after

    def _retrieve_by_time_range(
        self,
        chat_name: str,
        before_ts: datetime | None,
        after_ts: datetime | None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Fetch messages from a specific chat within a time window.

        ChromaDB metadata filtering on chat_name + timestamp range.
        Timestamps are stored as ISO-format strings; ChromaDB supports
        $gte / $lte on string fields for lexicographic comparison, which
        works correctly for ISO 8601 timestamps.

        Args:
            chat_name: Required — which chat to query
            before_ts: Only messages BEFORE this time (exclusive)
            after_ts:  Only messages AFTER this time (exclusive)
            limit:     Max messages to return

        Returns:
            List of dicts sorted by timestamp (newest first for before,
            oldest first for after).
        """
        if self.collection.count() == 0:
            return []

        chat_condition = {"chat_name": {"$eq": _bound_filter(chat_name)}}
        conditions = [chat_condition]

        if before_ts is not None:
            conditions.append(
                {"timestamp": {"$lt": before_ts.isoformat()}}
            )
        if after_ts is not None:
            conditions.append(
                {"timestamp": {"$gt": after_ts.isoformat()}}
            )

        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        try:
            result = self.collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        if not result or not result["documents"]:
            return []

        docs  = result["documents"]
        metas = result["metadatas"]

        paired = list(zip(docs, metas))

        if before_ts is not None:
            # Want messages closest to anchor (newest first → reverse later)
            paired.sort(key=lambda p: p[1].get("timestamp", ""), reverse=True)
        else:
            # Want messages closest to anchor (oldest first)
            paired.sort(key=lambda p: p[1].get("timestamp", ""))

        return [
            {
                "content":    doc,
                "sender":     meta.get("sender", ""),
                "chat_name":  meta.get("chat_name", ""),
                "timestamp":  meta.get("timestamp", ""),
                "platform":   meta.get("platform", ""),
                "distance":   0.0,
                "similarity": 0.0,
            }
            for doc, meta in paired[:limit]
        ]

    # ── Language-aware retrieval (from fork: Feature 6) ─────────────────────────

    def retrieve_similar_lang_aware(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
        sender_filter: str | None = None,
        chat_filter: str | None = None,
        prefer_language: str = "english",
    ) -> list[dict]:
        """
        Retrieve candidates, then re-rank preferring same-language matches.

        Strategy: over-fetch 3× top_k, score each result by similarity
        weighted with a language bonus, then return the top_k highest
        composite scores.
        """
        from brain.generator.intent_sensors import detect_language

        fetch_k = max(top_k * 3, 10)
        candidates = self.retrieve_similar(
            text=text,
            top_k=fetch_k,
            sender_filter=sender_filter,
            chat_filter=chat_filter,
        )

        if not candidates or len(candidates) <= top_k:
            return candidates[:top_k]

        LANG_BONUS = 0.12
        scored: list[tuple[float, dict]] = []

        for cand in candidates:
            content = cand.get("content", "")
            cand_lang = detect_language(content if isinstance(content, str) else "")
            sim = cand.get("similarity", 0.0)

            composite = sim + (LANG_BONUS if cand_lang == prefer_language else 0.0)
            scored.append((composite, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cand for _, cand in scored[:top_k]]

    # ── Stats / maintenance ────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of embedded messages in the store."""
        return self.collection.count()

    def senders(self) -> list[str]:
        """Unique sender names currently in the store."""
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["metadatas"])
        seen: set[str] = set()
        for meta in result["metadatas"]:
            if "sender" in meta:
                seen.add(meta["sender"])
        return sorted(seen)

    def stats(self) -> dict:
        """Summary statistics about the store contents."""
        total = self.collection.count()
        if total == 0:
            return {"total": 0, "senders": [], "chroma_path": str(self.chroma_path)}
        return {
            "total":       total,
            "senders":     self.senders(),
            "chroma_path": str(self.chroma_path),
        }

    def clear(self) -> None:
        """Delete all messages and recreate the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_id(record: MessageRecord) -> str:
    """
    Deterministic unique ID for a MessageRecord.

    v2: includes chat_name in the hash to prevent cross-chat collisions.
    Two messages with the same sender/timestamp/content in different chats
    are distinct records and must have distinct IDs.

    MD5 is fine here — this is deduplication, not security.
    """
    key = f"{record.timestamp.isoformat()}|{record.sender}|{record.chat_name}|{record.content[:200]}"
    return hashlib.md5(key.encode()).hexdigest()


def _make_metadata(record: MessageRecord) -> dict:
    """
    Flatten a MessageRecord into ChromaDB metadata.
    ChromaDB metadata values must be str, int, float, or bool.
    """
    return {
        "sender":     record.sender,
        "chat_name":  record.chat_name,
        "timestamp":  record.timestamp.isoformat(),
        "platform":   record.platform,
        "has_emoji":  record.has_emoji,
        "has_media":  record.has_media,
        "word_count": record.word_count,
        "char_count": record.char_count,
    }


def _build_where(
    sender_filter: str | None,
    chat_filter: str | None,
) -> dict | None:
    """
    Build a ChromaDB 'where' filter dict.

    Single:   {"sender": {"$eq": "Rishit"}}
    Multiple: {"$and": [{"sender": ...}, {"chat_name": ...}]}
    None:     no filtering

    Defense-in-depth: filter strings are capped at MAX_FILTER_CHARS so a
    pathological input cannot degrade query behaviour or push the model
    off-task.
    """
    conditions: list[dict] = []
    if sender_filter:
        conditions.append({"sender": {"$eq": _bound_filter(sender_filter)}})
    if chat_filter:
        conditions.append({"chat_name": {"$eq": _bound_filter(chat_filter)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


MAX_FILTER_CHARS = 256


def _bound_filter(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    return value[:MAX_FILTER_CHARS]