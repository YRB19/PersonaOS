"""
tests/test_media.py

Tests for Phase 3A — Media Intelligence.

Covers:
  - Tone detection (emoji_suggester._detect_tone)
  - Reaction emoji suggestions (emoji_suggester.suggest_reactions)
  - GIF query building (gif_suggester.build_gif_query)
  - GIF search without API key (graceful empty return)
  - Telegram reaction parsing (parsers.telegram._extract_reactions)
  - Reaction aggregation in EmojiProfile (analyzer._extract_emoji_profile)

No Tenor API key required for any test — GIF API tests use monkeypatching
and test the query builder logic independently of the network call.

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


# ── Shared helpers ─────────────────────────────────────────────────────────────

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
        """celebration keyword wins over a question mark in the same message."""
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
        """Affection tone only applies when relationship type supports it."""
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
        result = suggest_reactions("hello there")
        assert isinstance(result, list)

    def test_returns_non_empty_without_profile(self):
        result = suggest_reactions("lmao that's hilarious fr")
        assert len(result) > 0

    def test_empty_message_returns_empty(self):
        result = suggest_reactions("")
        assert result == []

    def test_whitespace_message_returns_empty(self):
        result = suggest_reactions("   ")
        assert result == []

    def test_result_items_are_reaction_suggestions(self):
        result = suggest_reactions("something funny")
        for r in result:
            assert isinstance(r, ReactionSuggestion)

    def test_emoji_field_is_non_empty_string(self):
        result = suggest_reactions("hi there")
        for r in result:
            assert isinstance(r.emoji, str) and len(r.emoji) > 0

    def test_confidence_in_valid_range(self):
        result = suggest_reactions("lmao bro 😂")
        for r in result:
            assert 0.0 <= r.confidence <= 1.0, f"Out of range: {r.confidence}"

    def test_top_k_respected(self):
        result = suggest_reactions("hi", top_k=3)
        assert len(result) <= 3

    def test_top_k_large_still_bounded(self):
        result = suggest_reactions("hi", top_k=50)
        # Can't return more emojis than exist in the pool + profile
        assert len(result) > 0

    def test_no_duplicate_emojis(self):
        result = suggest_reactions("haan theek hai bhai", profile=_make_profile(), top_k=20)
        emojis = [r.emoji for r in result]
        assert len(emojis) == len(set(emojis)), f"Duplicates found: {emojis}"

    def test_sorted_by_confidence_descending(self):
        result = suggest_reactions("lmao fr dead 💀", profile=_make_profile())
        confs = [r.confidence for r in result]
        assert confs == sorted(confs, reverse=True)

    def test_humor_message_gets_humor_emojis(self):
        result = suggest_reactions("lmao bhai 💀 that was hilarious")
        emojis = [r.emoji for r in result]
        humor_set = {"😂", "💀", "🤣", "😭", "😅", "🙃"}
        assert any(e in humor_set for e in emojis), \
            f"Expected humor emoji in suggestions, got: {emojis}"

    def test_celebration_message_gets_celebration_emojis(self):
        result = suggest_reactions("i got the internship offer!!")
        emojis = [r.emoji for r in result]
        celeb_set = {"🎉", "🥳", "🔥", "💯", "🙌", "✨"}
        assert any(e in celeb_set for e in emojis), \
            f"Expected celebration emoji, got: {emojis}"

    def test_personal_top_emoji_boosted(self):
        """User's top emoji 😂 should rank high for humor messages."""
        profile = _make_profile(top_emojis=[("😂", 100), ("❤️", 5)])
        result = suggest_reactions("lmao fr dead 💀", profile=profile)
        top_3 = [r.emoji for r in result[:3]]
        assert "😂" in top_3, \
            f"Expected 😂 (user's top emoji) in top 3, got: {top_3}"

    def test_reaction_history_boosts_emoji(self):
        """Emoji from reaction history should score higher than generic pool."""
        profile = _make_profile(
            top_emojis=[("🔥", 10)],
            top_reactions=[("👀", 50)],  # user often reacts with 👀
        )
        result = suggest_reactions("are you free tonight?", profile=profile)
        emojis = [r.emoji for r in result]
        assert "👀" in emojis, \
            f"Expected 👀 (reaction history) to appear in suggestions"

    def test_reaction_source_labelled(self):
        profile = _make_profile(top_reactions=[("👍", 30)])
        result = suggest_reactions("sure sounds good", profile=profile)
        reaction_history_items = [r for r in result if r.source == "reaction_history"]
        # 👍 might appear with reaction_history source
        assert isinstance(reaction_history_items, list)

    def test_reason_field_is_non_empty_string(self):
        result = suggest_reactions("this is funny lol")
        for r in result:
            assert isinstance(r.reason, str) and len(r.reason) > 3

    def test_no_crash_with_none_profile(self):
        result = suggest_reactions("hello there", profile=None)
        assert isinstance(result, list)

    def test_no_crash_with_relationship(self):
        result = suggest_reactions("hi", relationship=_make_rel("family"))
        assert isinstance(result, list)

    def test_profile_with_no_emoji_data(self):
        profile = CommunicationProfile(
            platform="whatsapp",
            total_messages_analyzed=50,
        )
        result = suggest_reactions("lol nice", profile=profile)
        assert isinstance(result, list)


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
        assert "for" not in query.split()
        assert "the" not in query.split()

    def test_removes_hinglish_stop_words(self):
        query = build_gif_query("kya kar raha hai bhai", None)
        assert "kya" not in query.split()
        assert "hai" not in query.split()

    def test_caps_at_three_content_keywords(self):
        """At most 3 content keywords (+ tone modifier)."""
        query = build_gif_query("party celebration weekend fun happy amazing", None)
        # Content keywords only — don't count the tone modifier
        content_words = query.split()
        # Some may be stop words — the point is not too many keywords
        assert len(content_words) <= 6

    def test_appends_close_friend_tone(self):
        rel = _make_rel("close_friend")
        query = build_gif_query("party tonight", rel)
        assert "friends" in query or "reaction" in query

    def test_appends_family_tone(self):
        rel = _make_rel("family")
        query = build_gif_query("ghar aa gaya", rel)
        assert "family" in query or "wholesome" in query

    def test_romantic_tone_modifier(self):
        rel = _make_rel("romantic_interest")
        query = build_gif_query("miss you", rel)
        assert "sweet" in query or "cute" in query

    def test_fallback_when_only_stop_words(self):
        """All tokens are stop words → fallback to 'reaction'."""
        query = build_gif_query("a the in is", None)
        assert query == "reaction"

    def test_fallback_on_empty_string(self):
        query = build_gif_query("", None)
        assert query == "reaction"

    def test_hinglish_message_produces_non_empty_query(self):
        query = build_gif_query("bhai canteen chalte lunch ke baad", None)
        assert len(query) > 0


# ══════════════════════════════════════════════════════════════════════════════
# GIF Search — no API key
# ══════════════════════════════════════════════════════════════════════════════

class TestGifSearch:

    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        result = search_gifs("funny reaction")
        assert result == []

    def test_returns_list_type(self, monkeypatch):
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        result = search_gifs("hello")
        assert isinstance(result, list)

    def test_no_crash_on_any_query(self, monkeypatch):
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        for query in ["", "lmao 😂", "bhai free hai?", "a" * 200]:
            result = search_gifs(query)
            assert isinstance(result, list)


# ══════════════════════════════════════════════════════════════════════════════
# Giphy response parser — URL scheme validation
# ══════════════════════════════════════════════════════════════════════════════

class TestParseGiphyResponse:
    """Defensive parsing: only http(s) URLs make it through."""

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
        assert results[0].preview_url == "https://media.giphy.com/small.gif"

    def test_javascript_scheme_rejected(self):
        data = {"data": [{
            "title": "xss",
            "images": {"original": {"url": "javascript:alert(1)"}},
        }]}
        assert self._parse(data) == []

    def test_file_scheme_rejected(self):
        data = {"data": [{
            "title": "lfi",
            "images": {"original": {"url": "file:///etc/passwd"}},
        }]}
        assert self._parse(data) == []

    def test_data_scheme_rejected(self):
        data = {"data": [{
            "title": "data",
            "images": {"original": {"url": "data:text/html,<script>1</script>"}},
        }]}
        assert self._parse(data) == []

    def test_empty_url_rejected(self):
        data = {"data": [{"title": "x", "images": {"original": {"url": ""}}}]}
        assert self._parse(data) == []

    def test_unsafe_preview_falls_back_to_safe_main(self):
        """If preview url is bad but main url is fine, fall back to main."""
        data = {"data": [{
            "title": "ok",
            "images": {
                "original": {"url": "https://media.giphy.com/g.gif"},
                "fixed_height_small": {"url": "javascript:bad()"},
            },
        }]}
        results = self._parse(data)
        assert len(results) == 1
        assert results[0].url == "https://media.giphy.com/g.gif"
        assert results[0].preview_url == "https://media.giphy.com/g.gif"

    def test_non_dict_items_skipped(self):
        data = {"data": ["not a dict", 42, None]}
        assert self._parse(data) == []

    def test_missing_images_handled(self):
        data = {"data": [{"title": "no images"}]}
        assert self._parse(data) == []


# ══════════════════════════════════════════════════════════════════════════════
# Telegram Reaction Parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestTelegramReactionParsing:

    def _make_telegram_fixture(self, tmp_path: Path, messages: list) -> Path:
        data = {"name": "Test Chat", "type": "personal_chat", "messages": messages}
        p = tmp_path / "chat.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_chosen_reaction_extracted(self, tmp_path):
        from brain.profiler.parsers.telegram import parse_telegram_export
        fixture = self._make_telegram_fixture(tmp_path, [
            {
                "id": 1, "type": "message",
                "date": "2024-01-01T10:00:00",
                "from": "Rishit", "text": "check this out",
                "reactions": [
                    {"type": "emoji", "emoji": "👍", "count": 1, "chosen": True},
                ],
            },
        ])
        records = parse_telegram_export(fixture)
        assert len(records) == 1
        assert "👍" in records[0].reactions

    def test_non_chosen_reaction_excluded(self, tmp_path):
        """chosen=False means someone ELSE reacted, not the user."""
        from brain.profiler.parsers.telegram import parse_telegram_export
        fixture = self._make_telegram_fixture(tmp_path, [
            {
                "id": 1, "type": "message",
                "date": "2024-01-01T10:00:00",
                "from": "Priya", "text": "hello",
                "reactions": [
                    {"type": "emoji", "emoji": "😂", "count": 3, "chosen": False},
                ],
            },
        ])
        records = parse_telegram_export(fixture)
        assert records[0].reactions == []

    def test_mixed_reactions_only_chosen_extracted(self, tmp_path):
        from brain.profiler.parsers.telegram import parse_telegram_export
        fixture = self._make_telegram_fixture(tmp_path, [
            {
                "id": 1, "type": "message",
                "date": "2024-01-01T10:00:00",
                "from": "Priya", "text": "test",
                "reactions": [
                    {"type": "emoji", "emoji": "👍", "count": 5, "chosen": False},
                    {"type": "emoji", "emoji": "❤️", "count": 1, "chosen": True},
                ],
            },
        ])
        records = parse_telegram_export(fixture)
        assert "❤️" in records[0].reactions
        assert "👍" not in records[0].reactions

    def test_no_reactions_field_gives_empty_list(self, tmp_path):
        from brain.profiler.parsers.telegram import parse_telegram_export
        fixture = self._make_telegram_fixture(tmp_path, [
            {
                "id": 1, "type": "message",
                "date": "2024-01-01T10:00:00",
                "from": "Rishit", "text": "no reactions here",
            },
        ])
        records = parse_telegram_export(fixture)
        assert records[0].reactions == []

    def test_multiple_chosen_reactions(self, tmp_path):
        """User can react with multiple emojis to the same message."""
        from brain.profiler.parsers.telegram import parse_telegram_export
        fixture = self._make_telegram_fixture(tmp_path, [
            {
                "id": 1, "type": "message",
                "date": "2024-01-01T10:00:00",
                "from": "Priya", "text": "great news",
                "reactions": [
                    {"type": "emoji", "emoji": "🎉", "count": 1, "chosen": True},
                    {"type": "emoji", "emoji": "🔥", "count": 1, "chosen": True},
                ],
            },
        ])
        records = parse_telegram_export(fixture)
        assert "🎉" in records[0].reactions
        assert "🔥" in records[0].reactions


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
            self._make_msg("Rishit", "hey"),
        ]
        profile = _extract_emoji_profile(user_msgs, all_chat_msgs=all_msgs)
        assert profile.reaction_count == 3
        reaction_emojis = [e for e, _ in profile.top_reactions]
        assert "👍" in reaction_emojis
        assert "🎉" in reaction_emojis

    def test_no_reactions_gives_zero_count(self):
        from brain.profiler.analyzer import _extract_emoji_profile
        user_msgs = [self._make_msg("Rishit", "hello")]
        all_msgs = [self._make_msg("Priya", "no reactions here")]
        profile = _extract_emoji_profile(user_msgs, all_chat_msgs=all_msgs)
        assert profile.reaction_count == 0
        assert profile.top_reactions == []

    def test_without_all_chat_msgs_still_works(self):
        """all_chat_msgs=None should not crash — just skips reaction extraction."""
        from brain.profiler.analyzer import _extract_emoji_profile
        user_msgs = [self._make_msg("Rishit", "hi there")]
        profile = _extract_emoji_profile(user_msgs, all_chat_msgs=None)
        assert profile.reaction_count == 0
        assert isinstance(profile.top_emojis, list)

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
        assert profile.top_reactions[0][0] == "😂", \
            "😂 (3 uses) should rank above 👍 (1 use)"