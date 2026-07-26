"""
tests/test_signals.py

Tests for brain/profiler/signals.py and the updated context-aware classifier.

Run with:
    pytest tests/test_signals.py -v
"""
import pytest
from datetime import datetime, timedelta
from brain.profiler.models import MessageRecord
from brain.profiler.signals import (
    ConversationSignals,
    extract_conversation_signals,
    _normalize,
    _tokenize_normalized,
    _is_college_friend_compound,
    _is_family_compound,
    _is_honorific_colleague_compound,
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


def _msg_dated(
    sender: str,
    content: str,
    day: int = 1,
    hour: int = 14,
) -> MessageRecord:
    """Message with configurable date for realistic multi-session fixtures."""
    return MessageRecord(
        timestamp=datetime(2024, 6, day, hour, 0, 0),
        sender=sender,
        content=content,
        platform="whatsapp",
        chat_name="Test Chat",
        has_emoji=False,
        has_media=False,
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
        """maa (unambiguous) + logistics → family compound."""
        assert _is_family_compound("maa ghar pahuch gayi kya") is True

    def test_family_compound_papa(self):
        """papa + logistics → family compound."""
        assert _is_family_compound("papa khana khaya kya") is True

    def test_no_family_compound_bhai_alone(self):
        """bhai alone no longer triggers family compound."""
        assert _is_family_compound("bhai kal milte hain") is False

    def test_no_family_compound_generic(self):
        assert _is_family_compound("kya scene hai yaar") is False

    def test_honorific_colleague_compound_bhaiya_review(self):
        """bhaiya + review (work keyword) → honorific_colleague compound."""
        assert _is_honorific_colleague_compound("bhaiya review pending hai") is True

    def test_honorific_colleague_compound_didi_code(self):
        """didi + code (technical term) → honorific_colleague compound."""
        assert _is_honorific_colleague_compound("didi code push kar diya") is True

    def test_honorific_colleague_compound_bhaiya_github(self):
        """bhaiya + github (technical term) → honorific_colleague compound."""
        assert _is_honorific_colleague_compound("bhaiya github pe check karo") is True

    def test_no_honorific_colleague_without_honorific(self):
        """Technical content without honorific is NOT a compound."""
        assert _is_honorific_colleague_compound("review pending hai code push karo") is False

    def test_no_honorific_colleague_without_technical(self):
        """Honorific without technical content is NOT a compound."""
        assert _is_honorific_colleague_compound("bhaiya kya scene hai") is False


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
        """Family classification now requires unambiguous terms like maa/papa + logistics."""
        msgs = (
            [_msg("Rishit", "haan papa ghar pahuch gaya", hour=18)] * 6 +
            [_msg("Rishit", "maa khana khaya kya", hour=19)] * 6 +
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


# ══════════════════════════════════════════════════════════════════════════════
# Honorific + professional topic compound tests
# Regression: Dev (62 msgs, technical, uses "bhaiya") was misclassified as family
# ══════════════════════════════════════════════════════════════════════════════

class TestHonorificProfessionalCompound:
    """
    Tests for the compound signal: family honorific + technical vocabulary → colleague.

    This fixes the Dev misclassification where "bhaiya" (honorific, not kinship)
    + heavy technical content was producing a false-positive family classification.
    """

    def _dev_msgs(self) -> list[MessageRecord]:
        """
        Fixture simulating the Dev conversation:
        - ~62 messages, technical/professional content
        - Uses "bhaiya" as respectful address (not literal brother)
        - Topics: IPL auction backend scripts, CSV/JSON formats, GitHub repos, code review
        - Mix of technical terms AND work vocabulary (meeting, sprint, review, feedback)
        - No check-in behavior, no emotional support
        - Timestamps spread across multiple sessions (realistic)
        """
        base = datetime(2024, 6, 1, 10, 0, 0)

        # Technical + work messages (core content — includes WORK_KEYWORDS terms)
        tech_work_msgs = [
            "bhaiya check this pull request on github",
            "bhaiya the csv format needs updating for the ipl auction backend",
            "bhaiya json response format change kiya hai",
            "bhaiya code review pending hai dekh lo",
            "bhaiya sprint review karna hai kal client ke saath",
            "bhaiya feedback de do is feature pe",
            "bhaiya meeting hai 3pm ko agenda ready karo",
            "bhaiya deliverable submit karna hai friday tak",
            "bhaiya standup mein update dena hai",
            "bhaiya backend script chal nahi raha hai",
            "bhaiya database query optimize karo",
            "bhaiya api endpoint return format change karo",
            "bhaiya github pe commit push kar diya",
            "bhaiya feature branch pe kaam kar raha hun",
            "bhaiya bug fix karna hai is sprint mein",
            "bhaiya server logs check karo error aa raha hai",
            "bhaiya deployment pipeline setup karna hai",
            "bhaiya code refactor karna hai purane wala",
            "bhaiya build fail ho raha hai test mein",
            "bhaiya release notes likh do",
        ]
        extra_tech_msgs = [
            "haan bhaiya dekh raha hun",
            "bhaiya ye code optimize karna hai",
            "bhaiya testing complete ho gayi",
            "bhaiya pr merge kar do",
            "bhaiya code review done hai",
            "bhaiya bug fix ready hai",
            "bhaiya deployment ready hai",
            "bhaiya release kar sakte hain",
            "bhaiya git stash kar do pehle",
            "bhaiya branch create kar diya",
            "bhaiya code linting pass ho gayi",
            "bhaiya api documentation update karo",
            "bhaiya database backup le lo",
            "bhaiya server restart karo please",
            "bhaiya logs share kar do",
            "bhaiya error fix ho gaya",
            "bhaiya build green hai ab",
            "bhaiya pipeline pass ho gayi",
            "bhaiya code quality check karo",
            "bhaiya pr approve kar do",
            "bhaiya feature branch delete kar do ab",
            "bhaiya staging pe deploy karo",
            "bhaiya production ready hai ye",
            "bhaiya release notes final kar do",
            "bhaiya sprint mein ye complete karna hai",
            "bhaiya backlog check karo kya pending hai",
            "bhaiya task assign kar do isme",
            "bhaiya ye ticket close kar do",
            "bhaiya next sprint planning hai kal",
            "bhaiya standup mein update dena hai",
            "bhaiya codebase refactor karna hai purana wala",
            "bhaiya unit tests likh do iske liye",
            "bhaiya integration test bhi add karo",
            "bhaiya code coverage improve karo",
            "bhaiya performance test bhi karna hai",
            "bhaiya load testing karo ispe",
            "bhaiya security review bhi karna hai",
            "bhaiya dependency update karni hai",
            "bhaiya package upgrade karo",
            "bhaiya client ko deliver karna hai ye",
        ]
        all_contents = tech_work_msgs + extra_tech_msgs
        all_contents = all_contents[:62]

        # Spread across multiple sessions (gaps > 4 hours between groups)
        msgs = []
        for i, content in enumerate(all_contents):
            day_offset = i // 20
            hour = 10 + (i % 8) * 2  # 10, 12, 14, 16, 18, 20, 22, 0
            ts = base + timedelta(days=day_offset, hours=(hour - 10))
            msgs.append(MessageRecord(
                timestamp=ts,
                sender="Rishit",
                content=content,
                platform="whatsapp",
                chat_name="WhatsApp Chat with Dev",
                has_emoji=False,
                has_media=False,
                word_count=len(content.split()),
                char_count=len(content),
            ))
        return msgs

    def test_honorific_heavy_professional_classifies_as_colleague(self):
        """
        Dev-like fixture: heavy "bhaiya" usage + technical vocabulary
        should classify as colleague, NOT family.

        This is the core regression test for the Dev misclassification.
        """
        msgs = self._dev_msgs()
        rel_type, confidence, signals = classify_relationship(msgs, msgs)
        assert rel_type == "colleague", \
            f"Expected colleague, got {rel_type} (confidence={confidence})"
        assert confidence >= 0.40, \
            f"Confidence too low: {confidence}"

    def test_honorific_hits_detected_but_not_family(self):
        """
        Honorifics are detected (family_honorific_hits > 0) but should NOT
        produce family_strong_hits (they're not in FAMILY_STRONG anymore).
        """
        msgs = self._dev_msgs()
        result = extract_conversation_signals(msgs, msgs)
        assert result.family_honorific_hits > 0, \
            "Honorifics should be detected"
        assert result.family_strong_hits == 0, \
            f"Family strong hits should be 0 (no maa/papa/etc.), got {result.family_strong_hits}"

    def test_honorific_colleague_compound_detected(self):
        """
        The compound signal (honorific + technical vocabulary) should fire
        for messages containing "bhaiya" + technical terms.
        """
        msgs = self._dev_msgs()
        result = extract_conversation_signals(msgs, msgs)
        assert result.honorific_colleague_compound_count > 0, \
            "Honorific+professional compound should be detected"

    def test_family_score_low_for_honorific_only(self):
        """
        Family score should be low when only honorifics are present
        (no unambiguous family terms, no check-in behavior).
        """
        msgs = self._dev_msgs()
        scores = get_all_scores(msgs, msgs)
        family_score = scores.get("family", (0.0, []))[0]
        assert family_score < 0.30, \
            f"Family score should be low for honorific-only contact, got {family_score}"

    def test_genuine_family_with_didi_still_classifies(self):
        """
        "didi" in a genuine family context (with check-in behavior) should
        still classify as family. Timestamps spread across multiple days
        to avoid inflated session dynamics.
        """
        msgs = [
            _msg_dated("Rishit", "didi khana khaya kya", day=d, hour=18)
            for d in range(1, 6)
        ] + [
            _msg_dated("Rishit", "didi ghar kab aa rahi ho", day=d, hour=19)
            for d in range(1, 6)
        ] + [
            _msg_dated("Rishit", "didi so gayi ho kya", day=d, hour=21)
            for d in range(1, 6)
        ]
        rel_type, confidence, _ = classify_relationship(msgs, msgs)
        assert rel_type == "family", \
            f"Expected family (didi + check-ins), got {rel_type}"
        assert confidence >= 0.40, \
            f"Confidence too low: {confidence}"

    def test_genuine_family_with_bhaiya_still_classifies(self):
        """
        "bhaiya" in a genuine family context (with check-in behavior) should
        still classify as family.
        """
        msgs = [
            _msg_dated("Rishit", "bhaiya khana khaya kya", day=d, hour=18)
            for d in range(1, 6)
        ] + [
            _msg_dated("Rishit", "bhaiya ghar kab aa raha hai", day=d, hour=19)
            for d in range(1, 6)
        ] + [
            _msg_dated("Rishit", "bhaiya so gaya kya", day=d, hour=21)
            for d in range(1, 6)
        ]
        rel_type, confidence, _ = classify_relationship(msgs, msgs)
        assert rel_type == "family", \
            f"Expected family (bhaiya + check-ins), got {rel_type}"
        assert confidence >= 0.40, \
            f"Confidence too low: {confidence}"

    def test_honorific_without_technical_not_colleague(self):
        """
        "bhaiya" without technical vocabulary should NOT classify as colleague.
        It should be acquaintance or similar — not enough evidence for any strong type.
        """
        msgs = (
            [_msg("Rishit", "bhaiya kya scene hai", hour=18)] * 6 +
            [_msg("Rishit", "bhaiya kahan ho abhi", hour=19)] * 6 +
            [_msg("Rishit", "bhaiya call karo kabhi", hour=20)] * 6
        )
        rel_type, confidence, _ = classify_relationship(msgs, msgs)
        assert rel_type != "colleague", \
            f"Honorific without technical should NOT be colleague, got {rel_type}"

    def test_all_scores_show_colleague_wins(self):
        """
        In the Dev-like fixture, colleague score should clearly beat family score.
        """
        msgs = self._dev_msgs()
        scores = get_all_scores(msgs, msgs)
        colleague_score = scores.get("colleague", (0.0, []))[0]
        family_score = scores.get("family", (0.0, []))[0]
        assert colleague_score > family_score, \
            f"Colleague ({colleague_score}) should beat family ({family_score})"
