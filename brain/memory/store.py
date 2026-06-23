"""
brain/memory/store.py  — v2

Changes from v1:
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
    Build a ChromaDB where-filter dict.

    Single:   {"sender": {"$eq": "Rishit"}}
    Multiple: {"$and": [{"sender": ...}, {"chat_name": ...}]}
    None:     no filtering
    """
    conditions: list[dict] = []
    if sender_filter:
        conditions.append({"sender": {"$eq": sender_filter}})
    if chat_filter:
        conditions.append({"chat_name": {"$eq": chat_filter}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}