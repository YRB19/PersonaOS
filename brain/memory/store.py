"""
brain/memory/store.py

PersonaStore — the memory layer of PersonaOS.

This is Phase 2A. It does two things:
  1. Store past messages as vector embeddings in ChromaDB
  2. Retrieve the most semantically similar past messages to any query

Why this matters:
  The reply generator (Phase 2B) needs examples of how YOU write in
  contexts similar to the incoming message. This store provides those
  examples. Without it, the generator has no basis for personalisation
  and falls back to generic AI output.

Data flow:
  add_messages(records)
    → filter embeddable messages
    → SentenceTransformer.encode() [local, ~90MB model, cached]
    → ChromaDB.upsert() [local SQLite file in data/db/chroma/]

  retrieve_similar(text, sender_filter="Rishit Babbar")
    → SentenceTransformer.encode(text)
    → ChromaDB.query() [cosine similarity search]
    → return top-k most similar past messages

Privacy: everything runs locally. No data leaves your machine.
"""
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from brain.profiler.models import MessageRecord


# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"  # ~470MB, fast on Apple Silicon
COLLECTION_NAME   = "persona_messages"
DEFAULT_TOP_K     = 5
BATCH_SIZE        = 128   # messages per SentenceTransformer call


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

        # SentenceTransformer downloads the model on first run (~90MB),
        # then caches it. Subsequent loads are instant.
        self.model = SentenceTransformer(model_name)

        # PersistentClient stores data on disk — survives process restarts.
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))

        # cosine distance is the standard for semantic similarity:
        # 0.0 = identical meaning, 1.0 = unrelated, 2.0 = opposite
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

        Args:
            records:       Parsed MessageRecord list (multi-sender OK)
            sender_filter: Only embed messages from this sender.
                           For PersonaOS, pass the user's own name —
                           you want to retrieve YOUR past messages,
                           not the other person's replies.

        Returns:
            Number of messages stored in this call.

        Notes:
            - Uses upsert: running this twice on the same export is safe.
              IDs are deterministic hashes — duplicates are overwritten, not doubled.
            - Skips media-only messages (nothing to embed in empty content).
            - Skips messages under 3 characters (reactions, single emoji, etc.)
            - Batches embedding calls for performance.
        """
        # Only embed messages we can actually learn from
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

        This is the core of RAG for PersonaOS. Given an incoming message
        you need to reply to, this returns your most similar past replies
        so the generator has concrete examples of your writing style in
        similar contexts.

        Args:
            text:          Incoming message or any query text
            top_k:         Number of results to return (default 5)
            sender_filter: Only retrieve messages from this sender
            chat_filter:   Only retrieve messages from this chat/contact

        Returns:
            List of dicts sorted by similarity (most similar first):
            [
                {
                    "content":    str,    # the past message text
                    "sender":     str,
                    "chat_name":  str,
                    "timestamp":  str,    # ISO 8601
                    "platform":   str,
                    "distance":   float,  # 0.0=identical, 2.0=opposite
                    "similarity": float,  # 1 - distance (higher = more similar)
                },
                ...
            ]

        Example:
            results = store.retrieve_similar(
                "are you free tomorrow?",
                top_k=5,
                sender_filter="Rishit Babbar"
            )
            # Returns 5 things you've said in the past that are
            # semantically closest to "are you free tomorrow?"
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

        # Cap top_k at collection size to avoid ChromaDB errors
        safe_top_k = min(top_k, self.collection.count())

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=safe_top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # where filter might exceed collection size — return empty gracefully
            return []

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        output = [
            {
                "content":   doc,
                "sender":    meta.get("sender", ""),
                "chat_name": meta.get("chat_name", ""),
                "timestamp": meta.get("timestamp", ""),
                "platform":  meta.get("platform", ""),
                "distance":  round(dist, 4),
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
        """
        Delete all messages and recreate the collection.
        Use this to rebuild the index from scratch after major changes.
        """
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_id(record: MessageRecord) -> str:
    """
    Deterministic unique ID for a MessageRecord.

    MD5 hash of (timestamp|sender|content[:200]).
    Same message → same ID → upsert overwrites, never duplicates.
    MD5 is fine here — this is deduplication, not security.
    """
    key = f"{record.timestamp.isoformat()}|{record.sender}|{record.content[:200]}"
    return hashlib.md5(key.encode()).hexdigest()


def _make_metadata(record: MessageRecord) -> dict:
    """
    Flatten a MessageRecord into ChromaDB metadata.

    ChromaDB metadata values must be str, int, float, or bool.
    datetime → ISO string. Everything else passes through.
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

    ChromaDB syntax:
      Single:   {"sender": {"$eq": "Rishit"}}
      Multiple: {"$and": [{"sender": ...}, {"chat_name": ...}]}
      None:     no filtering (return all)
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