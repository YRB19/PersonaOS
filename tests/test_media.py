"""
tests/test_media.py

Tests for Phase 3A — Media Intelligence.

Covers:
  - Tone detection (emoji_suggester._detect_tone)
  - Reaction emoji suggestions (emoji_suggester.suggest_reactions)
  - GIF query building (gif_suggester.build_gif_query)
  - GIF search without API key (graceful empty return)
  - Giphy response parser URL scheme validation
  - Telegram reaction parsing
  - Reaction aggregation in EmojiProfile

No Giphy API key required — GIF tests use monkeypatching and test
the query builder logic independently of network calls.

Run:
    pytest tests/test_media.py -v
"""
import json
import pytest
from datetime import datetime
from pathlib import Path

from brain.media.models import ReactionSuggestion, GifResult, MediaSuggestion
from brain.media.emoji_suggester import suggest_reactions, _detect_tone, _score_candidates
from brain.media.gif_suggester import build_gif_query, search_gifs
from brain.profiler.models import (
    CommunicationProfile, RelationshipProfile, EmojiProfile, StyleProfile,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_emoji_profile(
    top_emojis: list | None = None,
    top_reactions: list | None = None,
) -> EmojiProfile:
    return EmojiProfile(
        total_emoji_count=120,
        unique_emoji_count=10,
        top_emojis=top_emojis or [("😂", 50), ("🔥", 25), ("💀", 18), ("👍", 10)],
        emoji_frequency=0.42,
        reaction_count=len(top_reactions) if top_reactions else 0,
        top_reactions=top_reactions or [],
    )


def _make_profile(
    top_emojis: list | None = None,
    top_reactions: list | None = None,
) -> CommunicationProfile:
    return CommunicationProfile(
        platform="whatsapp",
        total_messages_analyzed=500,
        global_style=StyleProfile(),
        global_emoji=_make_emoji_profile(top_emojis, top_reactions),
    )


def _make_rel(rel_type: str = "close_friend") -> RelationshipProfile:
    return RelationshipProfile(
        contact_name="TestContact",
        message_count=150,
        relationship_type=rel_type,
        relationship_confidence=0.82,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tone Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestToneDetection:

    def test_humor_from_keyword(self):
        assert _detect_tone("lmao that was so funny bro") == "humor"

    def test_humor_from_emoji_in_text(self):
        assert _detect_tone("bhai 😂😂 died laughing") == "humor"

    def test_celebration_selected(self):
        assert _detect_tone("i got selected for the internship!!") == "celebration"

    def test_celebration_over_question(self):
        assert _detect_tone("you got selected??") == "celebration"

    def test_emotional_distress(self):
        assert _detect_tone("yaar tension ho rahi hai bahut") == "emotional"

    def test_question_from_punctuation(self):
        assert _detect_tone("are you free tonight?") == "question"

    def test_question_from_kya(self):
        assert _detect_tone("kya ho raha hai") == "question"

    def test_agreement_from_haan(self):
        assert _detect_tone("haan theek hai") == "agreement"

    def test_affection_with_close_friend_rel(self):
        rel = _make_rel("close_friend")
        assert _detect_tone("miss you yaar", relationship=rel) == "affection"

    def test_affection_skipped_without_matching_rel(self):
        rel = _make_rel("colleague")
        tone = _detect_tone("miss you", relationship=rel)
        assert tone != "affection"

    def test_support_from_help_keyword(self):
        assert _detect_tone("bhai help kar de stuck hu") == "support"

    def test_neutral_fallback(self):
        tone = _detect_tone("okay cool thanks")
        assert isinstance(tone, str) and len(tone) > 0

    def test_empty_message_returns_string(self):
        tone = _detect_tone("")
        assert isinstance(tone, str)


# ══════════════════════════════════════════════════════════════════════════════
# Reaction Emoji Suggestions
# ══════════════════════════════════════════════════════════════════════════════

class TestEmojiSuggester:

    def test_returns_list(self):
        assert isinstance(suggest_reactions("hello there"), list)

    def test_returns_non_empty_without_profile(self):
        assert len(suggest_reactions("lmao that's hilarious fr")) > 0

    def test_empty_message_returns_empty(self):
        assert suggest_reactions("") == []

    def test_whitespace_message_returns_empty(self):
        assert suggest_reactions("   ") == []

    def test_result_items_are_reaction_suggestions(self):
        for r in suggest_reactions("something funny"):
            assert isinstance(r, ReactionSuggestion)

    def test_emoji_field_is_non_empty_string(self):
        for r in suggest_reactions("hi there"):
            assert isinstance(r.emoji, str) and len(r.emoji) > 0

    def test_confidence_in_valid_range(self):
        for r in suggest_reactions("lmao bro 😂"):
            assert 0.0 <= r.confidence <= 1.0

    def test_top_k_respected(self):
        assert len(suggest_reactions("hi", top_k=3)) <= 3

    def test_no_duplicate_emojis(self):
        result = suggest_reactions("haan theek hai bhai", profile=_make_profile(), top_k=20)
        emojis = [r.emoji for r in result]
        assert len(emojis) == len(set(emojis))

    def test_sorted_by_confidence_descending(self):
        result = suggest_reactions("lmao fr dead 💀", profile=_make_profile())
        confs = [r.confidence for r in result]
        assert confs == sorted(confs, reverse=True)

    def test_humor_message_gets_humor_emojis(self):
        result = suggest_reactions("lmao bhai 💀 that was hilarious")
        emojis = [r.emoji for r in result]
        humor_set = {"😂", "💀", "🤣", "😭", "😅", "🙃"}
        assert any(e in humor_set for e in emojis)

    def test_celebration_message_gets_celebration_emojis(self):
        result = suggest_reactions("i got the internship offer!!")
        emojis = [r.emoji for r in result]
        celeb_set = {"🎉", "🥳", "🔥", "💯", "🙌", "✨"}
        assert any(e in celeb_set for e in emojis)

    def test_personal_top_emoji_boosted(self):
        profile = _make_profile(top_emojis=[("😂", 100), ("❤️", 5)])
        result = suggest_reactions("lmao fr dead 💀", profile=profile)
        top_3 = [r.emoji for r in result[:3]]
        assert "😂" in top_3

    def test_reaction_history_boosts_emoji(self):
        profile = _make_profile(
            top_emojis=[("🔥", 10)],
            top_reactions=[("👀", 50)],
        )
        result = suggest_reactions("are you free tonight?", profile=profile)
        emojis = [r.emoji for r in result]
        assert "👀" in emojis

    def test_reason_field_is_non_empty_string(self):
        for r in suggest_reactions("this is funny lol"):
            assert isinstance(r.reason, str) and len(r.reason) > 3

    def test_no_crash_with_none_profile(self):
        assert isinstance(suggest_reactions("hello there", profile=None), list)

    def test_no_crash_with_relationship(self):
        assert isinstance(suggest_reactions("hi", relationship=_make_rel("family")), list)


# ══════════════════════════════════════════════════════════════════════════════
# GIF Query Builder
# ══════════════════════════════════════════════════════════════════════════════

class TestGifQueryBuilder:

    def test_returns_string(self):
        result = build_gif_query("what are you doing tonight", None)
        assert isinstance(result, str) and len(result) > 0

    def test_removes_english_stop_words(self):
        query = build_gif_query("are you free for the party tonight", None)
        assert "are" not in query.split()
        assert "the" not in query.split()

    def test_removes_hinglish_stop_words(self):
        query = build_gif_query("kya kar raha hai bhai", None)
        assert "kya" not in query.split()
        assert "hai" not in query.split()

    def test_appends_close_friend_tone(self):
        query = build_gif_query("party tonight", _make_rel("close_friend"))
        assert "friends" in query or "reaction" in query

    def test_appends_family_tone(self):
        query = build_gif_query("ghar aa gaya", _make_rel("family"))
        assert "family" in query or "wholesome" in query

    def test_romantic_tone_modifier(self):
        query = build_gif_query("miss you", _make_rel("romantic_interest"))
        assert "sweet" in query or "cute" in query

    def test_fallback_when_only_stop_words(self):
        assert build_gif_query("a the in is", None) == "reaction"

    def test_fallback_on_empty_string(self):
        assert build_gif_query("", None) == "reaction"

    def test_hinglish_message_produces_non_empty_query(self):
        assert len(build_gif_query("bhai canteen chalte lunch ke baad", None)) > 0


# ══════════════════════════════════════════════════════════════════════════════
# GIF Search — no API key
# ══════════════════════════════════════════════════════════════════════════════

class TestGifSearch:

    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        assert search_gifs("funny reaction") == []

    def test_returns_list_type(self, monkeypatch):
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        assert isinstance(search_gifs("hello"), list)

    def test_no_crash_on_any_query(self, monkeypatch):
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        for query in ["", "lmao 😂", "bhai free hai?", "a" * 200]:
            assert isinstance(search_gifs(query), list)


# ══════════════════════════════════════════════════════════════════════════════
# Giphy response parser — URL scheme validation
# ══════════════════════════════════════════════════════════════════════════════

class TestParseGiphyResponse:

    def _parse(self, data):
        from brain.media.gif_suggester import _parse_giphy_response
        return _parse_giphy_response(data)

    def test_normal_https_urls_accepted(self):
        data = {"data": [{
            "title": "ok",
            "images": {
                "original": {"url": "https://media.giphy.com/g.gif"},
                "fixed_height_small": {"url": "https://media.giphy.com/small.gif"},
            },
        }]}
        results = self._parse(data)
        assert len(results) == 1
        assert results[0].url == "https://media.giphy.com/g.gif"
        assert results[0].source == "giphy"

    def test_javascript_scheme_rejected(self):
        data = {"data": [{"title": "xss", "images": {"original": {"url": "javascript:alert(1)"}}}]}
        assert self._parse(data) == []

    def test_file_scheme_rejected(self):
        data = {"data": [{"title": "lfi", "images": {"original": {"url": "file:///etc/passwd"}}}]}
        assert self._parse(data) == []

    def test_empty_url_rejected(self):
        data = {"data": [{"title": "x", "images": {"original": {"url": ""}}}]}
        assert self._parse(data) == []

    def test_unsafe_preview_falls_back_to_safe_main(self):
        data = {"data": [{
            "title": "ok",
            "images": {
                "original": {"url": "https://media.giphy.com/g.gif"},
                "fixed_height_small": {"url": "javascript:bad()"},
            },
        }]}
        results = self._parse(data)
        assert len(results) == 1
        assert results[0].preview_url == "https://media.giphy.com/g.gif"

    def test_non_dict_items_skipped(self):
        assert self._parse({"data": ["not a dict", 42, None]}) == []

    def test_missing_images_handled(self):
        assert self._parse({"data": [{"title": "no images"}]}) == []


# ══════════════════════════════════════════════════════════════════════════════
# Telegram Reaction Parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestTelegramReactionParsing:

    def _make_fixture(self, tmp_path: Path, messages: list) -> Path:
        data = {"name": "Test", "type": "personal_chat", "messages": messages}
        p = tmp_path / "chat.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_chosen_reaction_extracted(self, tmp_path):
        from brain.profiler.parsers.telegram import parse_telegram_export
        f = self._make_fixture(tmp_path, [{
            "id": 1, "type": "message",
            "date": "2024-01-01T10:00:00",
            "from": "Rishit", "text": "check this out",
            "reactions": [{"type": "emoji", "emoji": "👍", "count": 1, "chosen": True}],
        }])
        records = parse_telegram_export(f)
        assert "👍" in records[0].reactions

    def test_non_chosen_reaction_excluded(self, tmp_path):
        from brain.profiler.parsers.telegram import parse_telegram_export
        f = self._make_fixture(tmp_path, [{
            "id": 1, "type": "message",
            "date": "2024-01-01T10:00:00",
            "from": "Priya", "text": "hello",
            "reactions": [{"type": "emoji", "emoji": "😂", "count": 3, "chosen": False}],
        }])
        records = parse_telegram_export(f)
        assert records[0].reactions == []

    def test_no_reactions_field_gives_empty_list(self, tmp_path):
        from brain.profiler.parsers.telegram import parse_telegram_export
        f = self._make_fixture(tmp_path, [{
            "id": 1, "type": "message",
            "date": "2024-01-01T10:00:00",
            "from": "Rishit", "text": "no reactions here",
        }])
        records = parse_telegram_export(f)
        assert records[0].reactions == []


# ══════════════════════════════════════════════════════════════════════════════
# Reaction Aggregation in EmojiProfile
# ══════════════════════════════════════════════════════════════════════════════

class TestReactionAggregation:

    def _make_msg(self, sender, content, reactions=None):
        from brain.profiler.models import MessageRecord
        return MessageRecord(
            timestamp=datetime(2024, 1, 1, 14, 0, 0),
            sender=sender, content=content,
            platform="telegram", chat_name="Test",
            word_count=len(content.split()), char_count=len(content),
            reactions=reactions or [],
        )

    def test_reactions_aggregated_into_emoji_profile(self):
        from brain.profiler.analyzer import _extract_emoji_profile
        user_msgs = [self._make_msg("Rishit", "hey")]
        all_msgs = [
            self._make_msg("Priya", "check this", reactions=["👍"]),
            self._make_msg("Priya", "great news", reactions=["🎉", "🔥"]),
        ]
        profile = _extract_emoji_profile(user_msgs, all_chat_msgs=all_msgs)
        assert profile.reaction_count == 3
        reaction_emojis = [e for e, _ in profile.top_reactions]
        assert "👍" in reaction_emojis

    def test_no_reactions_gives_zero_count(self):
        from brain.profiler.analyzer import _extract_emoji_profile
        user_msgs = [self._make_msg("Rishit", "hello")]
        all_msgs = [self._make_msg("Priya", "no reactions")]
        profile = _extract_emoji_profile(user_msgs, all_chat_msgs=all_msgs)
        assert profile.reaction_count == 0

    def test_top_reactions_ordered_by_frequency(self):
        from brain.profiler.analyzer import _extract_emoji_profile
        user_msgs = [self._make_msg("Rishit", "hey")]
        all_msgs = [
            self._make_msg("Priya", "msg", reactions=["😂"]),
            self._make_msg("Priya", "msg", reactions=["😂"]),
            self._make_msg("Priya", "msg", reactions=["😂"]),
            self._make_msg("Priya", "msg", reactions=["👍"]),
        ]
        profile = _extract_emoji_profile(user_msgs, all_chat_msgs=all_msgs)
        assert profile.top_reactions[0][0] == "😂"
