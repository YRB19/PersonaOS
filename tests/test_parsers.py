"""
tests/test_parsers.py

Unit tests for PersonaOS chat parsers.
Run with: pytest tests/ -v

Tests use fixtures from tests/fixtures/ — no real personal data.
"""
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── WhatsApp Parser Tests ──────────────────────────────────────────────────────

class TestWhatsAppParser:

    def setup_method(self):
        """Import lazily so tests fail clearly if brain/ isn't importable."""
        from brain.profiler.parsers.whatsapp import (
            parse_whatsapp_export,
            _try_parse_line,
            _parse_timestamp,
            _has_emoji,
        )
        self.parse = parse_whatsapp_export
        self.try_parse_line = _try_parse_line
        self.parse_timestamp = _parse_timestamp
        self.has_emoji = _has_emoji

    def test_parse_sample_fixture(self):
        """Should parse the sample fixture without errors."""
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = self.parse(fixture, chat_name="Test Chat")
        assert len(records) > 0, "Expected at least one record"

    def test_skips_system_messages(self):
        """Should not include WhatsApp system messages."""
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = self.parse(fixture)
        contents = [r.content for r in records]
        assert not any("end-to-end encrypted" in c for c in contents)

    def test_media_messages_have_empty_content(self):
        """<Media omitted> messages should be marked has_media=True with empty content."""
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = self.parse(fixture)
        media = [r for r in records if r.has_media]
        assert len(media) > 0, "Expected at least one media message"
        for m in media:
            assert m.content == "", f"Media message should have empty content, got: {m.content!r}"

    def test_records_sorted_by_timestamp(self):
        """Records should be in chronological order."""
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = self.parse(fixture)
        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps), "Records should be sorted by timestamp"

    def test_word_and_char_counts(self):
        """Word and char counts should match content."""
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = self.parse(fixture)
        text_records = [r for r in records if r.content]
        for r in text_records[:10]:  # spot check first 10
            assert r.char_count == len(r.content), \
                f"char_count mismatch for: {r.content!r}"
            assert r.word_count == len(r.content.split()), \
                f"word_count mismatch for: {r.content!r}"

    def test_emoji_detection(self):
        """has_emoji should be True for messages with emoji."""
        assert self.has_emoji("hello 😂") is True
        assert self.has_emoji("hello world") is False
        assert self.has_emoji("🔥🔥🔥") is True
        assert self.has_emoji("no emoji here!") is False

    def test_platform_is_whatsapp(self):
        """All records from WhatsApp export should have platform='whatsapp'."""
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = self.parse(fixture)
        assert all(r.platform == "whatsapp" for r in records)

    def test_chat_name_override(self):
        """chat_name argument should override filename-based default."""
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = self.parse(fixture, chat_name="My Friends Group")
        assert all(r.chat_name == "My Friends Group" for r in records)

    def test_timestamp_formats(self):
        """Should parse multiple regional timestamp formats."""
        # International format
        result = self.parse_timestamp("01/01/2024", "10:01:00")
        assert result is not None
        assert result.year == 2024

        # Without seconds
        result = self.parse_timestamp("15/06/2023", "14:30")
        assert result is not None
        assert result.hour == 14

    def test_try_parse_line_bracketed_format(self):
        """Should parse [DD/MM/YYYY, HH:MM:SS] Sender: Message format."""
        line = "[01/01/2024, 10:01:00] Rishit: hey what's up"
        result = self.try_parse_line(line)
        assert result is not None
        assert result["sender"] == "Rishit"
        assert result["content"] == "hey what's up"

    def test_try_parse_line_dash_format(self):
        """Should parse DD/MM/YYYY, HH:MM - Sender: Message format."""
        line = "01/01/2024, 10:01 - Rishit: hey"
        result = self.try_parse_line(line)
        assert result is not None
        assert result["sender"] == "Rishit"

    def test_try_parse_line_returns_none_for_continuation(self):
        """Continuation lines (no timestamp) should return None."""
        result = self.try_parse_line("this is a continuation line")
        assert result is None


# ── Telegram Parser Tests ──────────────────────────────────────────────────────

class TestTelegramParser:

    def setup_method(self):
        from brain.profiler.parsers.telegram import (
            parse_telegram_export,
            _extract_text,
            _has_media,
        )
        self.parse = parse_telegram_export
        self.extract_text = _extract_text
        self.has_media = _has_media

    def test_parse_sample_fixture(self, tmp_path):
        """Should parse a minimal JSON fixture."""
        fixture = _create_minimal_telegram_fixture(tmp_path)
        records = self.parse(fixture)
        assert len(records) == 2

    def test_skips_service_messages(self, tmp_path):
        """Should skip messages with type='service'."""
        fixture = _create_telegram_fixture_with_service(tmp_path)
        records = self.parse(fixture)
        text_records = [r for r in records if r.content]
        senders = [r.sender for r in records]
        assert "Telegram" not in senders

    def test_extract_text_plain_string(self):
        """Plain string text should pass through unchanged."""
        msg = {"text": "hello world"}
        assert self.extract_text(msg) == "hello world"

    def test_extract_text_entity_list(self):
        """Entity list text should be reconstructed into plain string."""
        msg = {
            "text": [
                {"type": "plain", "text": "check "},
                {"type": "link", "text": "this link"},
                {"type": "plain", "text": " out"},
            ]
        }
        assert self.extract_text(msg) == "check this link out"

    def test_extract_text_empty(self):
        """Missing text field should return empty string."""
        assert self.extract_text({}) == ""

    def test_media_detection(self):
        """Messages with photo/video/sticker should be flagged as media."""
        assert self.has_media({"photo": "some_path.jpg"}) is True
        assert self.has_media({"sticker": "sticker.webp"}) is True
        assert self.has_media({"text": "plain message"}) is False
        assert self.has_media({}) is False

    def test_platform_is_telegram(self, tmp_path):
        """All records should have platform='telegram'."""
        fixture = _create_minimal_telegram_fixture(tmp_path)
        records = self.parse(fixture)
        assert all(r.platform == "telegram" for r in records)

    def test_records_sorted_by_timestamp(self, tmp_path):
        """Records should be in chronological order."""
        fixture = _create_minimal_telegram_fixture(tmp_path)
        records = self.parse(fixture)
        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps)


# ── Analyzer Tests ─────────────────────────────────────────────────────────────

class TestAnalyzer:

    def setup_method(self):
        from brain.profiler.analyzer import build_profile
        self.build_profile = build_profile

    def test_build_profile_from_fixture(self):
        """Should build a CommunicationProfile from the sample fixture."""
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = parse_whatsapp_export(fixture)
        profile = self.build_profile(records, user_name="Rishit")
        assert profile.total_messages_analyzed > 0
        assert profile.global_style is not None
        assert profile.platform == "whatsapp"

    def test_raises_on_unknown_user(self):
        """Should raise ValueError if user_name not in records."""
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = parse_whatsapp_export(fixture)
        with pytest.raises(ValueError, match="No messages found"):
            self.build_profile(records, user_name="NonExistentPerson")

    def test_raises_on_empty_records(self):
        """Should raise ValueError on empty record list."""
        with pytest.raises(ValueError, match="No records provided"):
            self.build_profile([], user_name="Anyone")

    def test_style_profile_computed(self):
        """StyleProfile should have numeric fields populated."""
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = parse_whatsapp_export(fixture)
        profile = self.build_profile(records, user_name="Rishit")
        style = profile.global_style
        assert style.avg_message_length > 0
        assert style.avg_word_count > 0


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _create_minimal_telegram_fixture(tmp_path: Path) -> Path:
    import json
    data = {
        "name": "Test Chat",
        "type": "personal_chat",
        "messages": [
            {
                "id": 1,
                "type": "message",
                "date": "2024-01-01T10:00:00",
                "from": "Rishit",
                "from_id": "user1",
                "text": "hey what's up",
            },
            {
                "id": 2,
                "type": "message",
                "date": "2024-01-01T10:01:00",
                "from": "Priya",
                "from_id": "user2",
                "text": "not much, you?",
            },
        ],
    }
    path = tmp_path / "telegram_fixture.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _create_telegram_fixture_with_service(tmp_path: Path) -> Path:
    import json
    data = {
        "name": "Test Chat",
        "type": "personal_chat",
        "messages": [
            {
                "id": 1,
                "type": "service",
                "date": "2024-01-01T09:00:00",
                "actor": "Telegram",
                "action": "phone_call",
                "text": "",
            },
            {
                "id": 2,
                "type": "message",
                "date": "2024-01-01T10:00:00",
                "from": "Rishit",
                "from_id": "user1",
                "text": "hello",
            },
        ],
    }
    path = tmp_path / "telegram_service_fixture.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ── Relationship Classifier Tests ──────────────────────────────────────────────

from datetime import datetime
from brain.profiler.models import MessageRecord


def _make_msg(sender: str, content: str, hour: int = 14, has_emoji: bool = False) -> MessageRecord:
    """Helper: build a MessageRecord without needing a full export."""
    return MessageRecord(
        timestamp=datetime(2024, 1, 1, hour, 0, 0),
        sender=sender,
        content=content,
        platform="whatsapp",
        chat_name="Test Chat",
        has_emoji=has_emoji,
        has_media=False,
        word_count=len(content.split()),
        char_count=len(content),
    )


class TestRelationshipClassifier:

    def setup_method(self):
        from brain.profiler.classifier import classify_relationship
        self.classify = classify_relationship

    def test_close_friend_signals(self):
        """High slang + late night + Hinglish → close_friend."""
        msgs = (
            [_make_msg("Rishit", "oye yaar kya ho raha", hour=23)] * 8
            + [_make_msg("Rishit", "lmao bro that's hilarious fr", hour=1)] * 7
            + [_make_msg("Rishit", "haan theek hai kal milte", hour=0)] * 5
        )
        rel_type, confidence, signals = self.classify(msgs, msgs)
        assert rel_type == "close_friend", f"Expected close_friend, got {rel_type}"
        assert confidence > 0.4
        assert len(signals) > 0

    def test_academic_peer_signals(self):
        """Academic keywords + file sharing → academic_peer."""
        msgs = (
            [_make_msg("Rishit", "bhai assignment submit kar diya?", hour=14)] * 5
            + [_make_msg("Rishit", "deadline kal hai yaar", hour=15)] * 5
            + [_make_msg("Rishit", "check newtonschool playground link", hour=16)] * 5
        )
        rel_type, confidence, signals = self.classify(msgs, msgs)
        assert rel_type in ("academic_peer", "close_friend"), \
            f"Expected academic_peer or close_friend, got {rel_type}"

    def test_family_signals(self):
        """Family terms → family classification."""
        msgs = (
            [_make_msg("Rishit", "haan papa aa raha hun", hour=18)] * 6
            + [_make_msg("Rishit", "maa khaana ready hai kya", hour=19)] * 6
            + [_make_msg("Rishit", "bhai ghar pe hai tu", hour=20)] * 6
        )
        rel_type, confidence, signals = self.classify(msgs, msgs)
        assert rel_type == "family", f"Expected family, got {rel_type}"
        assert confidence >= 0.4

    def test_insufficient_messages_returns_unknown(self):
        """Fewer than MIN_MESSAGES should return unknown."""
        msgs = [_make_msg("Rishit", "hello", hour=10)] * 3
        rel_type, confidence, signals = self.classify(msgs, msgs)
        assert rel_type == "unknown"
        assert confidence == 0.0

    def test_signals_are_human_readable(self):
        """Signals should be non-empty strings, not codes."""
        msgs = [_make_msg("Rishit", "oye yaar kya scene hai", hour=23)] * 15
        _, _, signals = self.classify(msgs, msgs)
        for signal in signals:
            assert isinstance(signal, str)
            assert len(signal) > 5, f"Signal too short: {signal!r}"

    def test_confidence_between_zero_and_one(self):
        """Confidence must always be in [0.0, 1.0]."""
        msgs = [_make_msg("Rishit", "hi how are you", hour=10)] * 12
        _, confidence, _ = self.classify(msgs, msgs)
        assert 0.0 <= confidence <= 1.0

    def test_relationship_type_is_valid(self):
        """Returned type must be one of the defined RELATIONSHIP_TYPES."""
        from brain.profiler.models import RELATIONSHIP_TYPES
        msgs = [_make_msg("Rishit", "oye sun bro", hour=22)] * 12
        rel_type, _, _ = self.classify(msgs, msgs)
        assert rel_type in RELATIONSHIP_TYPES, \
            f"Unknown type returned: {rel_type!r}"


# ── Multi-File Profile Tests ───────────────────────────────────────────────────

class TestMultiFileProfile:

    def setup_method(self):
        from brain.profiler.analyzer import build_profile
        self.build_profile = build_profile

    def test_sources_tracked_in_profile(self):
        """Profile should record which source files it was built from."""
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = parse_whatsapp_export(fixture)
        sources = ["sample_whatsapp.txt"]
        profile = self.build_profile(records, user_name="Rishit", sources=sources)
        assert profile.sources == sources

    def test_combining_records_from_two_parses(self):
        """
        Simulates running setup on two files:
        parse file1 + parse file2 → combine → build_profile.
        Message count should equal sum of both.
        """
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"

        records_a = parse_whatsapp_export(fixture)
        records_b = parse_whatsapp_export(fixture)  # same file twice for test

        combined = records_a + records_b
        profile = self.build_profile(
            combined,
            user_name="Rishit",
            sources=["chat_a.txt", "chat_b.txt"],
        )
        # Combined corpus has twice the messages — profile should reflect this
        your_msgs_a = [r for r in records_a if r.sender == "Rishit"]
        your_msgs_b = [r for r in records_b if r.sender == "Rishit"]
        expected_total = len(your_msgs_a) + len(your_msgs_b)
        assert profile.total_messages_analyzed == expected_total

    def test_empty_sources_list_acceptable(self):
        """sources=None should not crash — defaults to empty list."""
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = parse_whatsapp_export(fixture)
        profile = self.build_profile(records, user_name="Rishit", sources=None)
        assert profile.sources == []

    def test_relationships_have_type_after_build(self):
        """Every RelationshipProfile should have a relationship_type after build."""
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        fixture = FIXTURES_DIR / "sample_whatsapp.txt"
        records = parse_whatsapp_export(fixture)
        profile = self.build_profile(records, user_name="Rishit")
        for rel in profile.relationships:
            assert rel.relationship_type != "", \
                f"relationship_type empty for {rel.contact_name}"
            assert isinstance(rel.relationship_confidence, float)
            assert 0.0 <= rel.relationship_confidence <= 1.0


# ── profile_path security tests ───────────────────────────────────────────────

class TestProfilePathSecurity:
    """Regression tests for path-traversal in analyzer.profile_path()."""

    def setup_method(self):
        from brain.profiler.analyzer import profile_path
        self.profile_path = profile_path

    def test_normal_name_resolves_inside_profiles_dir(self, tmp_path):
        p = self.profile_path("Rishit Babbar", profiles_dir=tmp_path)
        assert p.parent == tmp_path
        assert p.name == "rishit_babbar.json"

    def test_path_traversal_dotdot_rejected(self, tmp_path):
        """'../../etc/passwd' must NOT escape the profiles directory."""
        with pytest.raises(ValueError):
            self.profile_path("../../etc/passwd", profiles_dir=tmp_path)

    def test_absolute_path_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            self.profile_path("/etc/passwd", profiles_dir=tmp_path)

    def test_windows_style_separators_rejected(self, tmp_path):
        """Backslash path components are rejected for safety (defensive)."""
        with pytest.raises(ValueError):
            self.profile_path("..\\..\\windows\\system32", profiles_dir=tmp_path)

    def test_empty_name_after_sanitise_rejected(self, tmp_path):
        """After stripping separators, an all-separator input must be rejected."""
        with pytest.raises(ValueError):
            self.profile_path("../../", profiles_dir=tmp_path)

    def test_case_insensitive_names_collide(self, tmp_path):
        """'Rishit Babbar' and 'RISHIT BABBAR' resolve to the same file."""
        a = self.profile_path("Rishit Babbar", profiles_dir=tmp_path)
        b = self.profile_path("RISHIT BABBAR", profiles_dir=tmp_path)
        assert a == b