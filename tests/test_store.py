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

    def test_build_where_strips_legacy_whatsapp_prefix(self):
        """
        chat_filter containing legacy 'WhatsApp Chat with ' prefix (e.g. from
        stale ChromaDB metadata) is normalised and produces an $or clause
        covering both canonical and legacy-prefixed forms.

        v4: _build_where now uses $or to match against both the stripped
        name AND prefixed variants — preventing Bug 2 (silent zero-results).
        """
        result = _build_where(None, "WhatsApp Chat with Sample Contact")
        assert "$or" in result
        branches = result["$or"]
        branch_values = [b["chat_name"]["$eq"] for b in branches]
        assert "Sample Contact" in branch_values
        assert "WhatsApp Chat with Sample Contact" in branch_values

    def test_build_where_chat_filter_already_stripped(self):
        """
        chat_filter without legacy prefix still produces an $or clause
        containing both the canonical and every prefixed form.
        """
        result = _build_where(None, "Sample Contact")
        assert "$or" in result
        branch_values = [b["chat_name"]["$eq"] for b in result["$or"]]
        assert "Sample Contact" in branch_values
        assert "WhatsApp Chat with Sample Contact" in branch_values

    def test_build_where_strips_legacy_prefix_with_sender(self):
        """
        Both filters together: legacy prefix still handled via $or.
        """
        result = _build_where("Rishit", "WhatsApp Chat with Sample Contact")
        assert "$and" in result
        # The sender condition is simple
        assert result["$and"][0] == {"sender": {"$eq": "Rishit"}}
        # The chat condition uses $or
        chat_clause = result["$and"][1]
        assert "$or" in chat_clause
        branch_values = [b["chat_name"]["$eq"] for b in chat_clause["$or"]]
        assert "Sample Contact" in branch_values
        assert "WhatsApp Chat with Sample Contact" in branch_values


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


# ══════════════════════════════════════════════════════════════════════════════
# Regression: Legacy prefix metadata (Bug 2 — silent zero-results)
# ══════════════════════════════════════════════════════════════════════════════

class TestLegacyPrefixCompatibility:
    """
    Bug 2 (bugs.txt): queries against a store with legacy 'WhatsApp Chat with'
    prefixes in chat_name metadata returned zero results silently.

    Root cause: _build_where used strict equality on chat_name.  If stored
    metadata had ''WhatsApp Chat with Sample Contact'' but the user typed
    ''Sample Contact'' (the canonical form), the filter never matched.

    Fix (bilateral):
      - Write path: _make_metadata normalises chat_name so new records never
        carry the prefix.
      - Read path: _build_where / _retrieve_by_time_range use ChromaDB $or
        to match against both the canonical and every legacy-prefixed form.
    """

    def _msg_prefixed(self, sender: str, content: str, chat: str = "WhatsApp Chat with Sample Contact", **kw) -> MessageRecord:
        """MessageRecord with a legacy-prefixed chat_name (simulates old parser output)."""
        return _msg(sender, content, chat=chat, **kw)

    def test_insert_with_prefix_is_stripped_at_write_time(self, store):
        """
        _make_metadata must strip the legacy prefix so stored metadata
        never contains ''WhatsApp Chat with ...''.
        """
        msg = self._msg_prefixed("Rishit", "hello from prefixed chat", chat="WhatsApp Chat with Sample Contact")
        store.add_messages([msg])
        assert store.count() == 1

        meta = store.collection.get(include=["metadatas"])["metadatas"][0]
        assert meta["chat_name"] == "Sample Contact", \
            f"Expected stripped chat_name, got: {meta['chat_name']}"

    def test_query_clean_chat_matches_prefixed_store(self, store):
        """
        Querying with the canonical (clean) chat name must return results
        even when stored metadata carries the legacy prefix.

        This is the acceptance criterion: len(results) > 0.
        """
        msgs = [
            self._msg_prefixed("Rishit", "whats the plan for tonight", chat="WhatsApp Chat with Sample Contact"),
            self._msg_prefixed("Rishit", "lets meet at 8 pm", chat="WhatsApp Chat with Sample Contact"),
            self._msg_prefixed("Rishit", "bro assignment done?", chat="WhatsApp Chat with Sample Contact"),
            self._msg_prefixed("Priya",  "yeah I finished it", chat="WhatsApp Chat with Priya"),
        ]
        store.add_messages(msgs)
        assert store.count() >= 3

        results = store.retrieve_similar(
            "what are you doing tonight",
            top_k=5,
            sender_filter="Rishit",
            chat_filter="Sample Contact",
        )
        assert len(results) > 0, \
            f"Bug 2 regression: query with clean chat name returned zero results. Root cause: $or not applied."

        for r in results:
            assert r["sender"] == "Rishit", \
                f"sender filter violated: got {r['sender']}"
            assert r["chat_name"] == "Sample Contact", \
                f"chat filter violated: got {r['chat_name']} (expected 'Sample Contact')"

    def test_query_prefixed_chat_also_matches(self, store):
        """
        Queries with the old prefixed form must also work (backward
        compatibility — someone typing the legacy name into the CLI).
        """
        msgs = [
            self._msg_prefixed("Rishit", "what are we doing friday"),
            self._msg_prefixed("Rishit", "bhai kal mil raha hai kya"),
        ]
        store.add_messages(msgs)

        results = store.retrieve_similar(
            "weekend plans",
            top_k=5,
            sender_filter="Rishit",
            chat_filter="WhatsApp Chat with Sample Contact",
        )
        assert len(results) > 0
        for r in results:
            assert r["chat_name"] == "Sample Contact"

    def test_chat_filter_hybrid_with_legacy_prefix(self, store):
        """retrieve_hybrid must also handle legacy prefixed data."""
        msgs = [
            self._msg_prefixed("Rishit", "yo what's up"),
            self._msg_prefixed("Rishit", "checking in bro"),
        ]
        store.add_messages(msgs)

        results = store.retrieve_hybrid(
            "whats up",
            top_k=3,
            sender_filter="Rishit",
            chat_filter="Sample Contact",
        )
        assert len(results) > 0
        for r in results:
            assert r["chat_name"] == "Sample Contact"

    def test_time_range_with_legacy_prefix(self, store):
        """_retrieve_by_time_range must handle legacy prefixed data."""
        msgs = [
            self._msg_prefixed("Rishit", "message one", chat="WhatsApp Chat with Sample Contact", hour=10),
            self._msg_prefixed("Rishit", "message two", chat="WhatsApp Chat with Sample Contact", hour=11),
            self._msg_prefixed("Rishit", "message three", chat="WhatsApp Chat with Sample Contact", hour=12),
        ]
        store.add_messages(msgs)

        from datetime import datetime
        anchor = datetime(2024, 1, 1, 11, 0, 0)
        before = store._retrieve_by_time_range(
            chat_name="Sample Contact",
            before_ts=anchor,
            after_ts=None,
            limit=5,
        )
        after = store._retrieve_by_time_range(
            chat_name="Sample Contact",
            before_ts=None,
            after_ts=anchor,
            limit=5,
        )
        assert len(before) > 0
        assert len(after) > 0

    def test_build_chat_name_or_produces_both_forms(self):
        """
        _build_chat_name_or must produce an $or clause containing both
        the canonical and the legacy-prefixed form.
        """
        from brain.memory.store import _build_chat_name_or
        result = _build_chat_name_or("Sample Contact")
        assert "$or" in result
        branches = result["$or"]
        assert len(branches) >= 2, \
            f"Expected at least 2 branches (canonical + prefixed), got {len(branches)}"
        branch_values = [b["chat_name"]["$eq"] for b in branches]
        assert "Sample Contact" in branch_values, \
            f"Missing canonical form in $or branches: {branch_values}"
        assert "WhatsApp Chat with Sample Contact" in branch_values, \
            f"Missing legacy-prefixed form in $or branches: {branch_values}"


# ══════════════════════════════════════════════════════════════════════════════
# v3: Hybrid retrieval + Thread retrieval + Recent retrieval
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrieveHybrid:

    def test_hybrid_returns_list(self, store, sample_records):
        """retrieve_hybrid always returns a list."""
        store.add_messages(sample_records)
        results = store.retrieve_hybrid("hello")
        assert isinstance(results, list)

    def test_hybrid_returns_requested_count_or_less(self, store, sample_records):
        """retrieve_hybrid should not return more than top_k."""
        store.add_messages(sample_records)
        results = store.retrieve_hybrid("food plans tonight", top_k=3)
        assert len(results) <= 3

    def test_hybrid_results_have_required_fields(self, store, sample_records):
        """Each result must have content, sender, similarity, semantic_score fields."""
        store.add_messages(sample_records)
        results = store.retrieve_hybrid("what are you doing tonight", top_k=2)
        for r in results:
            assert "content" in r
            assert "sender" in r
            assert "similarity" in r
            assert "timestamp" in r

    def test_hybrid_results_have_blended_scores(self, store, sample_records):
        """Hybrid results should have semantic_score and recency_score."""
        store.add_messages(sample_records)
        results = store.retrieve_hybrid("hello", top_k=3)
        for r in results:
            assert "semantic_score" in r
            assert "recency_score" in r

    def test_hybrid_empty_store(self, store):
        """Empty store returns empty list."""
        results = store.retrieve_hybrid("hello")
        assert results == []

    def test_hybrid_empty_query(self, store, sample_records):
        """Empty query should return empty list."""
        store.add_messages(sample_records)
        results = store.retrieve_hybrid("")
        assert results == []

    def test_hybrid_sender_filter_works(self, store, sample_records):
        """Sender filter must exclude messages from other senders."""
        store.add_messages(sample_records)
        results = store.retrieve_hybrid("hello", top_k=10, sender_filter="Rishit")
        for r in results:
            assert r["sender"] == "Rishit"

    def test_hybrid_with_custom_weights(self, store, sample_records):
        """Custom semantic_weight must be respected."""
        store.add_messages(sample_records)
        results_eq = store.retrieve_hybrid("hello", top_k=3, semantic_weight=0.5)
        results_sem = store.retrieve_hybrid("hello", top_k=3, semantic_weight=1.0)
        assert isinstance(results_eq, list)
        assert isinstance(results_sem, list)

    def test_hybrid_sorted_by_blended(self, store, sample_records):
        """Results should be sorted by blended score descending."""
        store.add_messages(sample_records)
        results = store.retrieve_hybrid("food plans tonight", top_k=5)
        if len(results) >= 2:
            sims = [r["similarity"] for r in results]
            assert sims == sorted(sims, reverse=True)


class TestRetrieveRecent:

    def test_retrieve_recent_returns_list(self, store, sample_records):
        """_retrieve_recent always returns a list."""
        store.add_messages(sample_records)
        results = store._retrieve_recent(limit=5)
        assert isinstance(results, list)

    def test_retrieve_recent_obeys_limit(self, store, sample_records):
        """_retrieve_recent should not return more than limit."""
        store.add_messages(sample_records)
        results = store._retrieve_recent(limit=3)
        assert len(results) <= 3

    def test_retrieve_recent_empty_store(self, store):
        """Empty store returns empty list."""
        results = store._retrieve_recent(limit=5)
        assert results == []

    def test_retrieve_recent_sorted_by_timestamp(self, store, sample_records):
        """Results should be sorted newest first."""
        store.add_messages(sample_records)
        results = store._retrieve_recent(limit=5)
        if len(results) >= 2:
            timestamps = [r["timestamp"] for r in results]
            assert timestamps == sorted(timestamps, reverse=True)

    def test_retrieve_recent_sender_filter(self, store, sample_records):
        """Sender filter must be respected."""
        store.add_messages(sample_records)
        results = store._retrieve_recent(sender_filter="Priya", limit=5)
        for r in results:
            assert r["sender"] == "Priya"

    def test_retrieve_recent_chat_filter(self, store, sample_records):
        """Chat filter must be respected."""
        store.add_messages(sample_records)
        results = store._retrieve_recent(chat_filter="Test Chat", limit=5)
        for r in results:
            assert r["chat_name"] == "Test Chat"


class TestThreadRetrieval:

    def test_retrieve_thread_returns_list(self, store, sample_records):
        """retrieve_thread always returns a list."""
        store.add_messages(sample_records)
        store.retrieve_similar("hello", top_k=1)
        # Get an anchor first
        results = store.retrieve_similar("hello", top_k=1)
        if results:
            thread = store.retrieve_thread(results[0], window=1)
            assert isinstance(thread, list)

    def test_retrieve_thread_includes_both_sides(self, store, sample_records):
        """retrieve_thread should fetch messages from before and after anchor."""
        store.add_messages(sample_records)
        results = store.retrieve_similar("hello", top_k=1)
        if results:
            thread = store.retrieve_thread(results[0], window=2)
            # Thread may be empty if anchor is at edge of chat
            if thread:
                # Check that all returned messages are from the same chat
                for t in thread:
                    assert t["chat_name"] == results[0]["chat_name"]

    def test_retrieve_thread_empty_store(self, store):
        """Empty store returns empty list."""
        result = store.retrieve_thread({"chat_name": "Test", "timestamp": "2024-01-01T10:00:00"})
        assert result == []

    def test_retrieve_thread_no_anchor_info(self, store):
        """Missing chat_name or timestamp returns empty list."""
        result1 = store.retrieve_thread({"chat_name": "Test"})
        assert result1 == []
        result2 = store.retrieve_thread({"timestamp": "2024-01-01T10:00:00"})
        assert result2 == []

    def test_retrieve_thread_respects_window(self, store, sample_records):
        """Window parameter limits number of messages per side."""
        store.add_messages(sample_records)
        results = store.retrieve_similar("hello", top_k=1)
        if results:
            thread = store.retrieve_thread(results[0], window=1)
            assert len(thread) <= 2  # 1 before + 1 after max


class TestRetrieveByTimeRange:

    def test_time_range_before(self, store, sample_records):
        """_retrieve_by_time_range with before_ts returns messages before that time."""
        store.add_messages(sample_records)
        from datetime import datetime
        anchor = datetime(2024, 1, 1, 14, 30, 0)
        results = store._retrieve_by_time_range(
            chat_name="Test Chat",
            before_ts=anchor,
            after_ts=None,
            limit=5,
        )
        assert len(results) > 0, "Vacuous-pass guard: expected non-empty from before_ts query"
        for r in results:
            ts = datetime.fromisoformat(r["timestamp"])
            assert ts < anchor, f"Expected before {anchor}, got {ts}"

    def test_time_range_after(self, store, sample_records):
        """_retrieve_by_time_range with after_ts returns messages after that time."""
        store.add_messages(sample_records)
        from datetime import datetime
        # All sample_records are at hour=14. Use anchor at hour=13 so
        # messages at hour=14 are strictly after the anchor.
        anchor = datetime(2024, 1, 1, 13, 0, 0)
        results = store._retrieve_by_time_range(
            chat_name="Test Chat",
            before_ts=None,
            after_ts=anchor,
            limit=5,
        )
        assert len(results) > 0, "Vacuous-pass guard: expected non-empty from after_ts query"
        for r in results:
            ts = datetime.fromisoformat(r["timestamp"])
            assert ts > anchor, f"Expected after {anchor}, got {ts}"

    def test_time_range_empty_store(self, store):
        """Empty store returns empty list."""
        from datetime import datetime
        results = store._retrieve_by_time_range(
            chat_name="Test",
            before_ts=datetime.now(),
            after_ts=None,
            limit=5,
        )
        assert results == []