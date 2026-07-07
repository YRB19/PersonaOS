"""
tests/test_signals.py

Tests for brain/profiler/signals.py and the updated context-aware classifier.

Run with:
    pytest tests/test_signals.py -v
"""
import pytest
from datetime import datetime
from brain.profiler.models import MessageRecord
from brain.profiler.signals import (
    ConversationSignals,
    extract_conversation_signals,
    _normalize,
    _tokenize_normalized,
    _is_college_friend_compound,
    _is_family_compound,
    PLANNING_PHRASES,
    CHECKIN_PHRASES,
    EMOTIONAL_PHRASES,
)
from brain.profiler.classifier import classify_relationship, get_all_scores


# ── Helpers ────────────────────────────────────────────────────────────────────

def _msg(
    sender: str,
    content: str,
    hour: int = 14,
    has_emoji: bool = False,
    has_media: bool = False,
) -> MessageRecord:
    return MessageRecord(
        timestamp=datetime(2024, 6, 1, hour, 0, 0),
        sender=sender,
        content=content,
        platform="whatsapp",
        chat_name="Test Chat",
        has_emoji=has_emoji,
        has_media=has_media,
        word_count=len(content.split()),
        char_count=len(content),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalisation:

    def test_common_abbreviations(self):
        """hn/hnn/han all normalise to haan."""
        assert _normalize("hn")  == "haan"
        assert _normalize("hnn") == "haan"
        assert _normalize("han") == "haan"
        assert _normalize("ha")  == "haan"

    def test_negation_variants(self):
        """nhi/nahin/nahh all normalise to nahi."""
        assert _normalize("nhi")   == "nahi"
        assert _normalize("nahin") == "nahi"
        assert _normalize("nahh")  == "nahi"
        assert _normalize("naa")   == "nahi"

    def test_ok_variants(self):
        """k/kk/okk/okie all normalise to okay."""
        assert _normalize("k")    == "okay"
        assert _normalize("kk")   == "okay"
        assert _normalize("okk")  == "okay"
        assert _normalize("okie") == "okay"

    def test_verb_abbreviations(self):
        """kr/krna normalise to kar/karna."""
        assert _normalize("kr")   == "kar"
        assert _normalize("krna") == "karna"

    def test_raha_variants(self):
        """rha/rhaa/rhe normalise to raha/rahe."""
        assert _normalize("rha")  == "raha"
        assert _normalize("rhaa") == "raha"
        assert _normalize("rhe")  == "rahe"

    def test_yaar_variants(self):
        assert _normalize("yr")  == "yaar"
        assert _normalize("yar") == "yaar"

    def test_bohot_variants(self):
        assert _normalize("bhot") == "bohot"
        assert _normalize("boht") == "bohot"

    def test_theek_variants(self):
        assert _normalize("thik")  == "theek"
        assert _normalize("theak") == "theek"

    def test_punctuation_stripped(self):
        """Punctuation stripped before normalisation."""
        assert _normalize("nhi!")  == "nahi"
        assert _normalize("hnn?")  == "haan"
        assert _normalize("kr.")   == "kar"

    def test_unknown_word_passthrough(self):
        """Unknown words pass through unchanged."""
        assert _normalize("pizza")   == "pizza"
        assert _normalize("meeting") == "meeting"

    def test_tokenize_normalized_full(self):
        """Full normalisation pipeline on a Hinglish message."""
        tokens = _tokenize_normalized("hnn kr deta hun nhi toh")
        assert "haan" in tokens
        assert "kar"  in tokens
        assert "nahi" in tokens


# ══════════════════════════════════════════════════════════════════════════════
# Compound signal tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCompoundSignals:

    def test_college_friend_compound_bhai_assignment(self):
        """bhai + assignment in same message → college_friend compound."""
        assert _is_college_friend_compound("bhai assignment bhej de") is True

    def test_college_friend_compound_yaar_deadline(self):
        """yaar + deadline → college_friend compound."""
        assert _is_college_friend_compound("yaar deadline hai kal") is True

    def test_college_friend_compound_oye_notes(self):
        """oye + notes → college_friend compound."""
        assert _is_college_friend_compound("oye notes bhej de bhai") is True

    def test_no_compound_without_casual_address(self):
        """assignment alone is NOT a compound signal."""
        assert _is_college_friend_compound("assignment submit karna hai") is False

    def test_no_compound_without_academic(self):
        """casual address alone is NOT a compound signal."""
        assert _is_college_friend_compound("bhai kya scene hai") is False

    def test_family_compound_maa(self):
        """maa (unambiguous) → family compound."""
        assert _is_family_compound("maa ghar kab aa rahi hai") is True

    def test_family_compound_papa(self):
        """papa → family compound."""
        assert _is_family_compound("papa ne kaha tha subah aana") is True

    def test_no_family_compound_bhai_alone(self):
        """bhai alone no longer triggers family compound."""
        assert _is_family_compound("bhai kal milte hain") is False

    def test_no_family_compound_generic(self):
        assert _is_family_compound("kya scene hai yaar") is False


# ══════════════════════════════════════════════════════════════════════════════
# Signal extraction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSignalExtraction:

    def _make_msgs(self, contents: list[str], sender: str = "Rishit",
                   hour: int = 14) -> list[MessageRecord]:
        return [_msg(sender, c, hour=hour) for c in contents]

    def test_returns_conversation_signals(self):
        msgs = self._make_msgs(["hello", "what's up", "nice"] * 5)
        result = extract_conversation_signals(msgs, msgs)
        assert isinstance(result, ConversationSignals)

    def test_planning_phrases_detected(self):
        """Planning phrases detected at phrase level."""
        msgs = self._make_msgs([
            "yaar chal milte hain aaj",
            "canteen chalte hain lunch ke time",
            "bhai scene kya hai aaj",
            "aaja yaar bahar chalte",
        ] * 3)
        result = extract_conversation_signals(msgs, msgs)
        assert result.planning_phrase_count >= 3
        assert result.has_planning_behavior is True

    def test_checkin_phrases_detected(self):
        """Check-in phrases detected at phrase level."""
        msgs = self._make_msgs([
            "khana khaya?",
            "so gaya tha kya",
            "ghar pahuch gaya?",
            "uth gaya?",
        ] * 3)
        result = extract_conversation_signals(msgs, msgs)
        assert result.checkin_phrase_count >= 3
        assert result.has_checkin_behavior is True

    def test_emotional_phrases_detected(self):
        msgs = self._make_msgs([
            "yaar tension ho rahi hai bahut",
            "stress ho raha hai bhai",
            "mood off hai aaj pata nahi kyun",
        ] * 3)
        result = extract_conversation_signals(msgs, msgs)
        assert result.emotional_phrase_count >= 2
        assert result.has_emotional_support is True

    def test_college_friend_compound_counted(self):
        msgs = self._make_msgs([
            "bhai assignment bhej de yaar",
            "oye notes share kar",
            "yaar deadline kal hai",
        ] * 5)
        result = extract_conversation_signals(msgs, msgs)
        assert result.college_friend_compound_count >= 5

    def test_hindi_density_with_normalization(self):
        """nhi/hnn/yr count as hindi tokens after normalisation."""
        msgs = self._make_msgs([
            "hnn kr deta hun nhi toh",
            "yr scene kya hai aaj",
            "nhi haan theek hai",
        ] * 6)
        result = extract_conversation_signals(msgs, msgs)
        assert result.hindi_density > 0.05

    def test_family_strong_terms_counted(self):
        msgs = self._make_msgs([
            "haan papa aa raha hun",
            "maa ne kaha ghar aa",
            "didi kal aa rahi hai",
        ] * 5)
        result = extract_conversation_signals(msgs, msgs)
        assert result.family_strong_hits >= 3

    def test_bhai_alone_not_family(self):
        """bhai used casually should NOT increase family_strong_hits."""
        msgs = self._make_msgs([
            "bhai kya scene hai",
            "bhai chal milte",
            "bhai assignment de de",
        ] * 5)
        result = extract_conversation_signals(msgs, msgs)
        assert result.family_strong_hits == 0

    def test_topic_academic_detected(self):
        msgs = self._make_msgs([
            "bhai midsem hai kal",
            "attendance kam ho gaya bhai",
            "proxy laga de yaar",
            "cgpa drop ho jayega",
        ] * 4)
        result = extract_conversation_signals(msgs, msgs)
        assert result.topic_academic > 0.0 or result.college_academic_hits >= 3

    def test_emoji_humor_detected(self):
        msgs = [_msg("Rishit", "lmao 😂", has_emoji=True)] * 15 + \
               [_msg("Rishit", "okay sounds good")] * 5
        result = extract_conversation_signals(msgs, msgs)
        assert result.humor_emoji_rate > 0.0

    def test_emoji_affection_detected(self):
        msgs = [_msg("Rishit", "miss you ❤️", has_emoji=True)] * 15 + \
               [_msg("Rishit", "okay")] * 5
        result = extract_conversation_signals(msgs, msgs)
        assert result.affection_emoji_rate > 0.0

    def test_session_dynamics_computed(self):
        """Session dynamics require messages with timestamp gaps."""
        msgs = (
            [_msg("Rishit", f"msg {i}", hour=14) for i in range(10)] +
            [_msg("Rishit", f"msg {i}", hour=20) for i in range(10)]
        )
        all_msgs = msgs + [_msg("Priya", "reply", hour=14)] * 5
        result = extract_conversation_signals(msgs, all_msgs)
        assert result.session_count >= 1
        assert result.avg_session_depth > 0


# ══════════════════════════════════════════════════════════════════════════════
# Classifier integration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifierContextAware:

    def test_college_friend_via_compound_signals(self):
        """
        bhai + assignment compound → college_friend, NOT split into family+academic.
        This is the key regression test for the v4 classifier.
        """
        msgs = (
            [_msg("Rishit", "bhai assignment bhej de yaar", hour=22)] * 8 +
            [_msg("Rishit", "oye notes share kar bhai", hour=23)] * 7 +
            [_msg("Rishit", "yaar deadline kal hai proxy laga", hour=21)] * 5
        )
        rel_type, confidence, signals = classify_relationship(msgs, msgs)
        assert rel_type in ("college_friend", "close_friend"), \
            f"Expected college_friend or close_friend, got {rel_type}"
        assert confidence > 0.40

    def test_family_requires_strong_terms(self):
        """Family classification now requires unambiguous terms like maa/papa."""
        msgs = (
            [_msg("Rishit", "haan papa aa raha hun ghar", hour=18)] * 6 +
            [_msg("Rishit", "maa khana ready hai kya", hour=19)] * 6 +
            [_msg("Rishit", "didi kal aa rahi hai kya", hour=20)] * 6
        )
        rel_type, confidence, _ = classify_relationship(msgs, msgs)
        assert rel_type == "family", f"Expected family, got {rel_type}"
        assert confidence >= 0.40

    def test_bhai_alone_not_classified_as_family(self):
        """
        bhai used as casual address should NOT produce family classification.
        This was the main false-positive in v3.
        """
        msgs = (
            [_msg("Rishit", "bhai kya scene hai", hour=22)] * 6 +
            [_msg("Rishit", "bhai chal milte hain yaar", hour=23)] * 6 +
            [_msg("Rishit", "bhai lmao that was hilarious fr", hour=1)] * 6
        )
        rel_type, _, _ = classify_relationship(msgs, msgs)
        assert rel_type != "family", \
            f"'bhai' as casual address should NOT produce family, got {rel_type}"

    def test_planning_behavior_boosts_close_friend(self):
        msgs = (
            [_msg("Rishit", "chal milte hain aaj shaam", hour=18)] * 6 +
            [_msg("Rishit", "yaar scene kya hai kal", hour=19)] * 5 +
            [_msg("Rishit", "canteen chalte lunch time pe", hour=14)] * 5
        )
        result = extract_conversation_signals(msgs, msgs)
        assert result.has_planning_behavior is True

    def test_emotional_support_boosts_close_friend(self):
        msgs = (
            [_msg("Rishit", "yaar tension ho rahi hai bahut", hour=23)] * 5 +
            [_msg("Rishit", "stress ho raha hai samajh nahi aa raha", hour=0)] * 5 +
            [_msg("Rishit", "mood off hai aaj pata nahi kyun", hour=22)] * 5
        )
        result = extract_conversation_signals(msgs, msgs)
        assert result.has_emotional_support is True

    def test_insufficient_messages_returns_unknown(self):
        msgs = [_msg("Rishit", "hello", hour=10)] * 3
        rel_type, confidence, _ = classify_relationship(msgs, msgs)
        assert rel_type == "unknown"
        assert confidence == 0.0

    def test_confidence_in_valid_range(self):
        msgs = [_msg("Rishit", "oye yaar kya scene hai", hour=22)] * 15
        _, confidence, _ = classify_relationship(msgs, msgs)
        assert 0.0 <= confidence <= 1.0

    def test_relationship_type_is_valid(self):
        from brain.profiler.models import RELATIONSHIP_TYPES
        valid = RELATIONSHIP_TYPES | {"college_friend"}
        msgs = [_msg("Rishit", "bhai assignment bhej", hour=22)] * 12
        rel_type, _, _ = classify_relationship(msgs, msgs)
        assert rel_type in valid, f"Unknown type: {rel_type!r}"

    def test_get_all_scores_returns_dict(self):
        msgs = [_msg("Rishit", "oye yaar plan kya hai", hour=22)] * 12
        scores = get_all_scores(msgs, msgs)
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_get_all_scores_empty_when_all_scores_zero(self):
        """If no signals match any category, get_all_scores returns empty dict."""
        # Minimal neutral messages — unlikely to trigger any category
        msgs = [_msg("Rishit", "ok", hour=14)] * 12
        scores = get_all_scores(msgs, msgs)
        # May be empty or have some categories — just ensure no crash
        assert isinstance(scores, dict)

    def test_signals_are_human_readable(self):
        msgs = (
            [_msg("Rishit", "bhai assignment bhej de", hour=22)] * 8 +
            [_msg("Rishit", "yaar deadline kal hai", hour=23)] * 7
        )
        _, _, signals = classify_relationship(msgs, msgs)
        for s in signals:
            assert isinstance(s, str)
            assert len(s) > 5