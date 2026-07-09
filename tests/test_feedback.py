"""
tests/test_feedback.py

Unit tests for brain/generator/feedback.py — FeedbackStore.

Run:
    pytest tests/test_feedback.py -v
"""
import pytest
from pathlib import Path
from brain.generator.feedback import FeedbackStore, FeedbackEntry
from brain.profiler.models import FeedbackSummary


class TestFeedbackEntry:

    def test_dataclass_fields(self):
        """FeedbackEntry must have all required fields."""
        entry = FeedbackEntry(
            incoming_message="hello",
            chosen_reply="hi back",
            rejected_replies=["no", "maybe"],
            contact_name="Priya",
            user_name="Rishit",
            timestamp="2024-01-01T12:00:00",
        )
        assert entry.incoming_message == "hello"
        assert entry.chosen_reply == "hi back"
        assert entry.rejected_replies == ["no", "maybe"]
        assert entry.contact_name == "Priya"
        assert entry.user_name == "Rishit"
        assert entry.timestamp == "2024-01-01T12:00:00"

    def test_relationship_type_default(self):
        """relationship_type defaults to 'unknown'."""
        entry = FeedbackEntry(
            incoming_message="hello",
            chosen_reply="hi",
            rejected_replies=[],
            contact_name="X",
            user_name="Y",
            timestamp="2024-01-01T12:00:00",
        )
        assert entry.relationship_type == "unknown"

    def test_custom_relationship_type(self):
        """Custom relationship_type must be stored."""
        entry = FeedbackEntry(
            incoming_message="hello",
            chosen_reply="hi",
            rejected_replies=[],
            contact_name="X",
            user_name="Y",
            timestamp="2024-01-01T12:00:00",
            relationship_type="close_friend",
        )
        assert entry.relationship_type == "close_friend"


class TestFeedbackStore:

    def test_record_returns_feedback_entry(self, tmp_path):
        """record() must return a FeedbackEntry."""
        fb = FeedbackStore(path=tmp_path / "fb")
        entry = fb.record(
            incoming="are you free?",
            chosen="haan yaar free hoon",
            rejected=["no", "maybe later"],
            contact="Priya",
            user="Rishit Babbar",
        )
        assert isinstance(entry, FeedbackEntry)
        assert entry.chosen_reply == "haan yaar free hoon"

    def test_record_writes_file(self, tmp_path):
        """record() must create a .jsonl file on disk."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi", [], "Priya", "Rishit Babbar")
        filepath = fb._user_path("Rishit Babbar")
        assert filepath.exists()
        content = filepath.read_text()
        assert "hello" in content
        assert "hi" in content

    def test_get_positive_examples_returns_list(self, tmp_path):
        """get_positive_examples must return a list."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi back", ["no"], "Priya", "Rishit Babbar")
        results = fb.get_positive_examples("Rishit Babbar")
        assert isinstance(results, list)

    def test_get_positive_examples_content(self, tmp_path):
        """get_positive_examples returns chosen replies."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi back", ["no"], "Priya", "Rishit Babbar")
        results = fb.get_positive_examples("Rishit Babbar")
        assert len(results) > 0
        assert results[0]["content"] == "hi back"
        assert results[0]["source"] == "feedback_positive"

    def test_get_positive_examples_filters_by_contact(self, tmp_path):
        """Contact filter must work for positive examples."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi", [], "Priya", "Rishit Babbar")
        fb.record("sup", "hey", [], "Someone Else", "Rishit Babbar")
        results = fb.get_positive_examples("Rishit Babbar", contact="Priya")
        for r in results:
            assert r["contact_name"] == "Priya"

    def test_get_negative_examples_returns_list(self, tmp_path):
        """get_negative_examples must return a list."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi", ["rejected1", "rejected2"], "Priya", "Rishit Babbar")
        results = fb.get_negative_examples("Rishit Babbar")
        assert isinstance(results, list)

    def test_get_negative_examples_content(self, tmp_path):
        """get_negative_examples returns rejected replies."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi", ["bad reply", "worse reply"], "Priya", "Rishit Babbar")
        results = fb.get_negative_examples("Rishit Babbar")
        assert len(results) > 0
        assert results[0]["source"] == "feedback_negative"

    def test_get_empty_when_no_data(self, tmp_path):
        """No data returns empty list for both positive and negative."""
        fb = FeedbackStore(path=tmp_path / "fb")
        assert fb.get_positive_examples("NonExistent") == []
        assert fb.get_negative_examples("NonExistent") == []

    def test_obey_limits(self, tmp_path):
        """Both get_positive and get_negative must respect limit parameter."""
        fb = FeedbackStore(path=tmp_path / "fb")
        for i in range(5):
            fb.record(f"incoming{i}", f"chosen{i}", [f"rejected{i}"], "Priya", "Rishit Babbar")
        pos = fb.get_positive_examples("Rishit Babbar", limit=2)
        assert len(pos) <= 2
        neg = fb.get_negative_examples("Rishit Babbar", limit=2)
        assert len(neg) <= 2

    def test_acceptance_rate_zero_on_no_data(self, tmp_path):
        """stats acceptance_rate must be 1.0 when no data (vacuously true)."""
        fb = FeedbackStore(path=tmp_path / "fb")
        s = fb.stats("NonExistent")
        assert s["acceptance_rate"] == 1.0

    def test_stats_structure(self, tmp_path):
        """stats must contain expected keys."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi", [], "Priya", "Rishit Babbar")
        s = fb.stats("Rishit Babbar")
        assert "total_recorded" in s
        assert "accepted_count" in s
        assert "rejected_count" in s
        assert "acceptance_rate" in s
        assert "top_accepted_patterns" in s

    def test_stats_counts(self, tmp_path):
        """stats must report correct counts."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi", ["rejected1", "rejected2"], "Priya", "Rishit Babbar")
        fb.record("sup", "hey", [], "Priya", "Rishit Babbar")
        s = fb.stats("Rishit Babbar")
        assert s["total_recorded"] == 2
        assert s["accepted_count"] == 2
        assert s["rejected_count"] == 2  # only first entry has 2 rejected

    def test_multiple_users_isolation(self, tmp_path):
        """Data from different users must not mix."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi", [], "Priya", "User A")
        fb.record("sup", "hey", [], "Priya", "User B")
        assert len(fb.get_positive_examples("User A")) == 1
        assert len(fb.get_positive_examples("User B")) == 1

    def test_appends_not_overwrites(self, tmp_path):
        """record() must append, not overwrite."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("msg1", "reply1", [], "Priya", "Rishit Babbar")
        fb.record("msg2", "reply2", [], "Priya", "Rishit Babbar")
        entries = fb._read_entries("Rishit Babbar")
        assert len(entries) == 2

    def test_record_with_rel_type(self, tmp_path):
        """record() with relationship type must be stored in the file."""
        fb = FeedbackStore(path=tmp_path / "fb")
        fb.record("hello", "hi", [], "Priya", "Rishit Babbar", rel_type="close_friend")
        entries = fb._read_entries("Rishit Babbar")
        assert len(entries) == 1
        assert entries[0]["relationship_type"] == "close_friend"


# ══════════════════════════════════════════════════════════════════════════════
# FeedbackSummary model (data contract)
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedbackSummary:

    def test_all_defaults(self):
        """FeedbackSummary must have sensible defaults."""
        s = FeedbackSummary()
        assert s.total_recorded == 0
        assert s.accepted_count == 0
        assert s.rejected_count == 0
        assert s.acceptance_rate == 0.0
        assert s.top_accepted_patterns == []

    def test_custom_values(self):
        """FeedbackSummary must accept custom values."""
        s = FeedbackSummary(
            total_recorded=10,
            accepted_count=8,
            rejected_count=2,
            acceptance_rate=0.8,
            top_accepted_patterns=["haan yaar", "okay sure"],
        )
        assert s.total_recorded == 10
        assert s.acceptance_rate == 0.8
        assert len(s.top_accepted_patterns) == 2

    def test_is_pydantic_model(self):
        """FeedbackSummary must be Pydantic v2 BaseModel."""
        from pydantic import BaseModel
        assert issubclass(FeedbackSummary, BaseModel)

    def test_acceptance_rate_float(self):
        """acceptance_rate must be a float."""
        s = FeedbackSummary(acceptance_rate=0.75)
        assert isinstance(s.acceptance_rate, float)
