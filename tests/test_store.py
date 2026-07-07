"""
tests/test_store.py

Unit tests for brain/memory/store.py (Phase 2A).

These tests use a real SentenceTransformer model and real ChromaDB —
no mocks. The first run downloads the embedding model (~90MB, cached
after that). Subsequent runs are fast.

Run with:
    pytest tests/test_store.py -v
    pytest tests/test_store.py -v -s    # shows print output
"""
import pytest
from datetime import datetime
from pathlib import Path
from brain.profiler.models import MessageRecord
from brain.memory.store import PersonaStore, _make_id, _make_metadata, _build_where


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _msg(sender: str, content: str, chat: str = "Test Chat", hour: int = 14) -> MessageRecord:
    """Helper: build a MessageRecord without a full export."""
    return MessageRecord(
        timestamp=datetime(2024, 1, 1, hour, 0, 0),
        sender=sender,
        content=content,
        platform="whatsapp",
        chat_name=chat,
        has_emoji=False,
        has_media=False,
        word_count=len(content.split()),
        char_count=len(content),
    )


@pytest.fixture
def store(tmp_path):
    """Fresh PersonaStore backed by a temp directory — deleted after each test."""
    return PersonaStore(chroma_path=tmp_path / "chroma_test")


@pytest.fixture
def sample_records():
    """A small set of realistic messages for testing."""
    return [
        _msg("Rishit", "yo what are you doing tonight"),
        _msg("Rishit", "wanna grab food later?"),
        _msg("Rishit", "lmao that's hilarious bro"),
        _msg("Rishit", "okay sounds good see you at 8"),
        _msg("Rishit", "the assignment is due tomorrow right"),
        _msg("Rishit", "can you send me the notes from class"),
        _msg("Rishit", "I think the deadline got extended"),
        _msg("Rishit", "haan yaar theek hai"),
        _msg("Priya",  "sure I'll be there"),
        _msg("Priya",  "yes deadline is tomorrow 11pm"),
    ]


# ── Helpers ────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_make_id_is_deterministic(self):
        """Same record always produces same ID."""
        r = _msg("Rishit", "hello world")
        assert _make_id(r) == _make_id(r)

    def test_make_id_differs_for_different_records(self):
        """Different content produces different ID."""
        r1 = _msg("Rishit", "hello world")
        r2 = _msg("Rishit", "goodbye world")
        assert _make_id(r1) != _make_id(r2)

    def test_make_id_is_string(self):
        """ID must be a string (ChromaDB requirement)."""
        r = _msg("Rishit", "test")
        assert isinstance(_make_id(r), str)

    def test_make_metadata_types(self):
        """All metadata values must be str/int/float/bool (ChromaDB requirement)."""
        r = _msg("Rishit", "hello")
        meta = _make_metadata(r)
        for key, val in meta.items():
            assert isinstance(val, (str, int, float, bool)), \
                f"Metadata key '{key}' has invalid type {type(val)}"

    def test_make_metadata_has_required_keys(self):
        """Metadata must include sender, chat_name, timestamp."""
        r = _msg("Rishit", "hello", chat="My Chat")
        meta = _make_metadata(r)
        assert meta["sender"] == "Rishit"
        assert meta["chat_name"] == "My Chat"
        assert "timestamp" in meta
        assert "platform" in meta

    def test_build_where_no_filters(self):
        """No filters → None (ChromaDB: no filter)."""
        assert _build_where(None, None) is None

    def test_build_where_sender_only(self):
        """Single sender filter."""
        result = _build_where("Rishit", None)
        assert result == {"sender": {"$eq": "Rishit"}}

    def test_build_where_both_filters(self):
        """Both filters → $and compound."""
        result = _build_where("Rishit", "Priya")
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_build_where_bounds_long_filters(self):
        """Pathologically long filter strings must be truncated."""
        long_sender = "X" * 10_000
        result = _build_where(long_sender, None)
        assert result == {"sender": {"$eq": "X" * 256}}
        assert len(result["sender"]["$eq"]) == 256

    def test_build_where_non_str_filter_coerced(self):
        """Non-string filters are coerced to strings, not crashed."""
        result = _build_where(123, None)  # type: ignore[arg-type]
        assert result == {"sender": {"$eq": "123"}}


# ── PersonaStore ───────────────────────────────────────────────────────────────

class TestPersonaStore:

    def test_store_initialises_empty(self, store):
        """Fresh store should have 0 messages."""
        assert store.count() == 0

    def test_add_messages_returns_count(self, store, sample_records):
        """add_messages returns number of messages stored."""
        n = store.add_messages(sample_records)
        assert n > 0
        assert n <= len(sample_records)

    def test_add_messages_skips_media(self, store):
        """Media-only messages should not be embedded."""
        media_msg = MessageRecord(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            sender="Rishit",
            content="",
            platform="whatsapp",
            chat_name="Test",
            has_media=True,
            word_count=0,
            char_count=0,
        )
        n = store.add_messages([media_msg])
        assert n == 0
        assert store.count() == 0

    def test_add_messages_skips_empty_content(self, store):
        """Empty content messages should be skipped."""
        empty = _msg("Rishit", "")
        n = store.add_messages([empty])
        assert n == 0

    def test_add_messages_with_sender_filter(self, store, sample_records):
        """Only messages from the filtered sender should be stored."""
        n = store.add_messages(sample_records, sender_filter="Rishit")
        rishit_msgs = [r for r in sample_records
                       if r.sender == "Rishit" and r.content.strip() and not r.has_media]
        assert n == len(rishit_msgs)

    def test_upsert_is_idempotent(self, store, sample_records):
        """Adding same messages twice should not increase count."""
        store.add_messages(sample_records)
        count_after_first = store.count()
        store.add_messages(sample_records)
        count_after_second = store.count()
        assert count_after_first == count_after_second

    def test_count_increases_after_add(self, store, sample_records):
        """Count should reflect stored messages."""
        assert store.count() == 0
        store.add_messages(sample_records)
        assert store.count() > 0

    def test_retrieve_returns_list(self, store, sample_records):
        """retrieve_similar always returns a list."""
        store.add_messages(sample_records)
        results = store.retrieve_similar("hello")
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self, store, sample_records):
        """retrieve_similar should not return more than top_k results."""
        store.add_messages(sample_records)
        results = store.retrieve_similar("food plans tonight", top_k=3)
        assert len(results) <= 3

    def test_retrieve_results_have_required_fields(self, store, sample_records):
        """Each result must have content, sender, similarity fields."""
        store.add_messages(sample_records)
        results = store.retrieve_similar("what are you doing tonight", top_k=2)
        for r in results:
            assert "content" in r
            assert "sender" in r
            assert "similarity" in r
            assert "distance" in r

    def test_retrieve_similarity_in_range(self, store, sample_records):
        """Similarity scores must be between 0 and 1."""
        store.add_messages(sample_records)
        results = store.retrieve_similar("doing something tonight")
        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0

    def test_retrieve_sorted_by_similarity(self, store, sample_records):
        """Results should be sorted highest similarity first."""
        store.add_messages(sample_records)
        results = store.retrieve_similar("assignment deadline tomorrow")
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_retrieve_semantic_relevance(self, store, sample_records):
        """Semantically similar messages should rank above unrelated ones."""
        store.add_messages(sample_records)
        results = store.retrieve_similar(
            "is the homework due soon",
            top_k=3,
            sender_filter="Rishit",
        )
        # At least one of the top results should be about assignments/deadline
        academic_keywords = {"assignment", "deadline", "notes", "class", "due"}
        top_contents = " ".join(r["content"].lower() for r in results)
        assert any(kw in top_contents for kw in academic_keywords), \
            f"Expected academic content in top results, got: {[r['content'] for r in results]}"

    def test_retrieve_sender_filter_works(self, store, sample_records):
        """Sender filter should exclude messages from other senders."""
        store.add_messages(sample_records)
        results = store.retrieve_similar(
            "sure sounds good",
            top_k=10,
            sender_filter="Rishit",
        )
        for r in results:
            assert r["sender"] == "Rishit", \
                f"Got message from unexpected sender: {r['sender']}"

    def test_retrieve_empty_store_returns_empty(self, store):
        """Querying an empty store should return empty list, not raise."""
        results = store.retrieve_similar("hello there")
        assert results == []

    def test_retrieve_empty_query_returns_empty(self, store, sample_records):
        """Empty query string should return empty list gracefully."""
        store.add_messages(sample_records)
        results = store.retrieve_similar("")
        assert results == []

    def test_senders_returns_list(self, store, sample_records):
        """senders() should return unique sender names."""
        store.add_messages(sample_records)
        s = store.senders()
        assert isinstance(s, list)
        assert "Rishit" in s

    def test_senders_empty_store(self, store):
        """senders() on empty store should return empty list."""
        assert store.senders() == []

    def test_stats_structure(self, store, sample_records):
        """stats() should return a dict with total and senders."""
        store.add_messages(sample_records)
        s = store.stats()
        assert "total" in s
        assert "senders" in s
        assert s["total"] > 0

    def test_clear_empties_store(self, store, sample_records):
        """clear() should remove all messages."""
        store.add_messages(sample_records)
        assert store.count() > 0
        store.clear()
        assert store.count() == 0

    def test_add_after_clear(self, store, sample_records):
        """Should be able to add messages after clearing."""
        store.add_messages(sample_records)
        store.clear()
        n = store.add_messages(sample_records)
        assert n > 0
        assert store.count() > 0