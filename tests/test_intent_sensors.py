"""
tests/test_intent_sensors.py

Unit tests for brain/generator/intent_sensors.py — the heuristic sensor module
that powers features 3, 4, 5, 6, 7, 8.

All tests are pure Python, zero external dependencies, fast.
"""
import json

import pytest
from brain.generator.intent_sensors import (
    detect_intent,
    detect_emotion,
    detect_language,
    detect_stage,
    should_skip_reply,
    build_explainability,
    ExplainabilityRecord,
    ConversationContext,
    INTENT_GREETING,
    INTENT_QUESTION,
    INTENT_PLANNING,
    INTENT_TASK,
    INTENT_EMOTIONAL,
    INTENT_HUMOR,
    INTENT_GENERAL,
    EMOTION_HAPPY,
    EMOTION_SAD,
    EMOTION_ANGRY,
    EMOTION_NEUTRAL,
    EMOTION_EXCITED,
    STAGE_FIRST,
    STAGE_MID,
    STAGE_LEFT_READ,
    STAGE_ENDING,
    STAGE_UNKNOWN,
    LANG_HINGLISH,
    LANG_ENGLISH,
)


# ══════════════════════════════════════════════════════════════════════════════
# Canned LLM — keeps generate() tests offline
# ══════════════════════════════════════════════════════════════════════════════

_CANNED_DRAFTS = [
    "haan yaar chalte hain",
    "theek hai bro sounds good",
    "lol yes fr im free",
]
_CANNED_LLM_JSON = json.dumps({"replies": _CANNED_DRAFTS})


class _FakeOllama:
    """
    Stand-in for the ollama module's generate().

    ReplyGenerator._call_ollama does `import ollama` inside the function body and
    calls ollama.generate(...), so patching the attribute on the real module is
    enough — no import-time patching required. `calls` lets a test assert the LLM
    was reached (or deliberately skipped).
    """

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        # _call_ollama reads .response off a dict or a response object
        return {"response": _CANNED_LLM_JSON}


@pytest.fixture
def fake_ollama(monkeypatch):
    """Replace ollama.generate with the canned response for one test."""
    ollama = pytest.importorskip("ollama")
    fake = _FakeOllama()
    monkeypatch.setattr(ollama, "generate", fake.generate)
    return fake


# ══════════════════════════════════════════════════════════════════════════════
# Intent detection (Feature 3)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentDetection:
    def test_greeting_short_hi(self):
        assert detect_intent("hi") == INTENT_GREETING

    def test_greeting_hey(self):
        assert detect_intent("heyy") == INTENT_GREETING

    def test_greeting_gm(self):
        assert detect_intent("good morning") == INTENT_GREETING

    def test_greeting_kaise_ho(self):
        assert detect_intent("kaise ho") == INTENT_GREETING

    def test_question_with_mark(self):
        assert detect_intent("are you free tonight?") == INTENT_QUESTION

    def test_question_kya_start(self):
        assert detect_intent("kya kal free hai") == INTENT_QUESTION

    def test_question_kaise(self):
        assert detect_intent("kaise jaana hai") == INTENT_QUESTION

    def test_task_assignment(self):
        assert detect_intent("bhai assignment bhej de deadline hai") == INTENT_TASK

    def test_task_project(self):
        assert detect_intent("project submit karna hai kal") == INTENT_TASK

    def test_planning_meetup(self):
        assert detect_intent("kal milte hain canteen") == INTENT_PLANNING

    def test_planning_scene(self):
        assert detect_intent("aaj ka scene kya hai") == INTENT_PLANNING

    def test_emotional_sad(self):
        assert detect_intent("yaar mood off hai aaj") == INTENT_EMOTIONAL
        assert detect_intent("tension ho rahi hai") == INTENT_EMOTIONAL

    def test_humor_lol(self):
        assert detect_intent("lol that was funny") == INTENT_HUMOR

    def test_humor_haha(self):
        assert detect_intent("haha mast tha") == INTENT_HUMOR

    def test_general_fallback(self):
        assert detect_intent("theek hai") == INTENT_GENERAL
        assert detect_intent("ok done hai") == INTENT_GENERAL

    def test_greeting_trumps_other(self):
        assert detect_intent("hii") == INTENT_GREETING

    def test_question_trumps_task(self):
        assert detect_intent("assignment bhej de?") == INTENT_QUESTION

    def test_empty_input(self):
        assert detect_intent("") == INTENT_GENERAL
        assert detect_intent("   ") == INTENT_GENERAL


# ══════════════════════════════════════════════════════════════════════════════
# Emotion detection (Feature 5)
# ══════════════════════════════════════════════════════════════════════════════

class TestEmotionDetection:
    def test_happy(self):
        assert detect_emotion("badhiya hai yaar maza aa gaya") == EMOTION_HAPPY

    def test_sad(self):
        assert detect_emotion("yaar tension ho rahi hai") == EMOTION_SAD
        assert detect_emotion("mood off 😢") == EMOTION_SAD

    def test_angry(self):
        assert detect_emotion("wtf this is so annoying 😡") == EMOTION_ANGRY

    def test_excited(self):
        assert detect_emotion("finally yess let's go!! 🔥") == EMOTION_EXCITED

    def test_excited_exclamation(self):
        assert detect_emotion("bro!! that was insane!!!") == EMOTION_EXCITED

    def test_neutral_fallback(self):
        assert detect_emotion("theek hai kal milte hain") == EMOTION_NEUTRAL
        assert detect_emotion("ok") == EMOTION_NEUTRAL

    def test_empty(self):
        assert detect_emotion("") == EMOTION_NEUTRAL


# ══════════════════════════════════════════════════════════════════════════════
# Conversation stage detection (Feature 4)
# ══════════════════════════════════════════════════════════════════════════════

class TestStageDetection:
    def test_first_message_long_gap(self):
        ctx = ConversationContext(hours_since_last=80.0)
        assert detect_stage("hey long time!", ctx) == STAGE_FIRST

    def test_left_on_read_with_unread(self):
        ctx = ConversationContext(hours_since_last=10.0, unread_count=2)
        assert detect_stage("aur batao", ctx) == STAGE_LEFT_READ

    def test_mid_conversation(self):
        ctx = ConversationContext(hours_since_last=1.0, unread_count=0)
        assert detect_stage("chal canteen", ctx) == STAGE_MID

    def test_ending_bye(self):
        ctx = ConversationContext(hours_since_last=1.0)
        assert detect_stage("bye", ctx) == STAGE_ENDING

    def test_ending_goodnight(self):
        ctx = ConversationContext(hours_since_last=2.0)
        assert detect_stage("goodnight", ctx) == STAGE_ENDING

    def test_ending_theek_hai_bye(self):
        ctx = ConversationContext(hours_since_last=1.0)
        assert detect_stage("theek hai bye", ctx) == STAGE_ENDING

    def test_unknown_no_context(self):
        assert detect_stage("hello", None) == STAGE_UNKNOWN

    def test_long_gap_no_unread_is_mid(self):
        ctx = ConversationContext(hours_since_last=50.0, unread_count=0)
        assert detect_stage("hey", ctx) == STAGE_MID


# ══════════════════════════════════════════════════════════════════════════════
# Language detection (Feature 6)
# ══════════════════════════════════════════════════════════════════════════════

class TestLanguageDetection:
    def test_pure_hindi_hinglish(self):
        assert detect_language("haan bhai main ghar pahuch gaya") == LANG_HINGLISH

    def test_hinglish_mixed(self):
        assert detect_language("yaar kal milte hain canteen") == LANG_HINGLISH

    def test_pure_english(self):
        assert detect_language("hey what are you doing tonight") == LANG_ENGLISH

    def test_mostly_english_returns_english(self):
        assert detect_language("I think we should meet at canteen tomorrow") == LANG_ENGLISH

    def test_empty(self):
        assert detect_language("") == LANG_ENGLISH


# ══════════════════════════════════════════════════════════════════════════════
# Smart silence (Feature 8)
# ══════════════════════════════════════════════════════════════════════════════

class TestSmartSilence:
    def test_ok_skipped(self):
        assert should_skip_reply("ok") is True

    def test_haan_skipped(self):
        assert should_skip_reply("haan") is True

    def test_theek_hai_skipped(self):
        assert should_skip_reply("theek hai") is True

    def test_k_skipped(self):
        assert should_skip_reply("k") is True
        assert should_skip_reply("kk") is True

    def test_emoji_only_skipped(self):
        assert should_skip_reply("👍") is True

    def test_normal_message_not_skipped(self):
        assert should_skip_reply("bhai free hai aaj shaam") is False
        assert should_skip_reply("are you coming tonight?") is False
        assert should_skip_reply("assignment bhej de deadline hai") is False

    def test_empty_skipped(self):
        assert should_skip_reply("") is True
        assert should_skip_reply("   ") is True

    def test_long_message_with_ok_not_skipped(self):
        assert should_skip_reply("ok kal milte hain theek hai") is False


# ══════════════════════════════════════════════════════════════════════════════
# Explainability record (Feature 7)
# ══════════════════════════════════════════════════════════════════════════════

class TestExplainability:
    def test_build_explainability_returns_record(self):
        examples = [
            {"content": "yo what are you doing tonight", "similarity": 0.92},
        ]
        rec = build_explainability(
            incoming="yo free?",
            intent=INTENT_QUESTION,
            emotion=EMOTION_NEUTRAL,
            stage=STAGE_MID,
            language=LANG_HINGLISH,
            relationship_type="close_friend",
            relationship_label="close friend",
            examples=examples,
            has_profile=True,
            has_relationship=True,
            silence_result=(False, ""),
        )
        assert isinstance(rec, ExplainabilityRecord)
        assert rec.intent == INTENT_QUESTION
        assert rec.has_profile is True
        assert rec.retrieved_examples_count == 1
        assert rec.silence_skip is False

    def test_to_dict_has_keys(self):
        rec = build_explainability(
            incoming="hi", intent=INTENT_GENERAL, emotion=EMOTION_NEUTRAL,
            stage=STAGE_MID, language=LANG_HINGLISH,
            relationship_type="close_friend", relationship_label="close friend",
            examples=[], has_profile=False, has_relationship=False,
            silence_result=(False, ""),
        )
        d = rec.to_dict()
        for k in ("intent", "emotion", "stage", "language",
                  "relationship_type", "relationship_label",
                  "has_profile", "has_relationship",
                  "retrieved_examples_count"):
            assert k in d, f"Missing key: {k}"

    def test_to_text_non_empty(self):
        rec = build_explainability(
            incoming="yo", intent=INTENT_GREETING, emotion=EMOTION_HAPPY,
            stage=STAGE_FIRST, language=LANG_ENGLISH,
            relationship_type="close_friend", relationship_label="close friend",
            examples=[
                {"content": "yo what are you doing tonight"},
                {"content": "haan yaar milte hain"},
            ],
            has_profile=True, has_relationship=True,
            silence_result=(False, ""),
        )
        text = rec.to_text()
        assert len(text) > 50
        assert "Intent:" in text
        assert "Why these drafts" in text
        assert "close friend" in text

    def test_silence_explainability(self):
        rec = build_explainability(
            incoming="ok", intent=INTENT_GENERAL, emotion=EMOTION_NEUTRAL,
            stage=STAGE_UNKNOWN, language=LANG_ENGLISH,
            relationship_type="unknown", relationship_label="friend",
            examples=[], has_profile=False, has_relationship=False,
            silence_result=(True, "low-value acknowledgment"),
        )
        assert rec.silence_skip is True
        assert "Skip reason" in rec.to_text()

    def test_example_previews_truncated(self):
        examples = [{"content": "X" * 500}] * 10
        rec = build_explainability(
            incoming="hi", intent=INTENT_GENERAL, emotion=EMOTION_NEUTRAL,
            stage=STAGE_UNKNOWN, language=LANG_ENGLISH,
            relationship_type="unknown", relationship_label="friend",
            examples=examples, has_profile=False, has_relationship=False,
            silence_result=(False, ""),
        )
        previews = rec.retrieved_example_previews
        assert len(previews) <= 5
        for preview in previews:
            assert len(preview) <= 80


# ══════════════════════════════════════════════════════════════════════════════
# Integration — prompt content with intent/emotion (Feature 3+5)
# ══════════════════════════════════════════════════════════════════════════════

class TestPromptV6:
    """build_prompt_v6 produces prompts with intent/stage/emotion guidance."""

    def test_question_intent_appears_in_system(self):
        from brain.generator.reply import build_prompt_v6
        from brain.generator.intent_sensors import INTENT_QUESTION
        system, _ = build_prompt_v6(
            "kal free hai?", "Priya", None, None, [],
            intent=INTENT_QUESTION,
        )
        assert "Question" in system or "question" in system.lower() or \
               "INTENT" in system

    def test_greeting_intent_produces_light_prompt(self):
        from brain.generator.reply import build_prompt_v6
        from brain.generator.intent_sensors import INTENT_GREETING
        system, _ = build_prompt_v6(
            "heyy", "Priya", None, None, [],
            intent=INTENT_GREETING,
        )
        assert "greeting" in system.lower() or "Greeting" in system

    def test_emotional_intent_includes_support_guidance(self):
        from brain.generator.reply import build_prompt_v6
        from brain.generator.intent_sensors import INTENT_EMOTIONAL
        system, _ = build_prompt_v6(
            "tension ho rahi hai", "Priya", None, None, [],
            intent=INTENT_EMOTIONAL,
        )
        assert "empathy" in system.lower() or "support" in system.lower() or \
               "emotional" in system.lower() or "validate" in system.lower()

    def test_sad_emotion_tone_override(self):
        from brain.generator.reply import build_prompt_v6
        from brain.generator.intent_sensors import (
            INTENT_EMOTIONAL, EMOTION_SAD,
        )
        system, _ = build_prompt_v6(
            "mood off hai yaar", "Priya", None, None, [],
            intent=INTENT_EMOTIONAL, emotion=EMOTION_SAD,
        )
        assert "sad" in system.lower() or "care" in system.lower() or \
               "gentle" in system.lower() or "tone down" in system.lower()

    def test_angry_emotion_deescalation(self):
        from brain.generator.reply import build_prompt_v6
        from brain.generator.intent_sensors import (
            INTENT_GENERAL, EMOTION_ANGRY,
        )
        system, _ = build_prompt_v6(
            "wtf gussa aa raha hai", "Priya", None, None, [],
            intent=INTENT_GENERAL, emotion=EMOTION_ANGRY,
        )
        assert "de-escal" in system.lower() or "calm" in system.lower() or \
               "resolve" in system.lower() or "validate" in system.lower()

    def test_stage_first_talks_about_gap(self):
        from brain.generator.reply import build_prompt_v6
        from brain.generator.intent_sensors import (
            INTENT_GENERAL, STAGE_FIRST,
        )
        system, _ = build_prompt_v6(
            "heyy", "Priya", None, None, [],
            intent=INTENT_GENERAL, stage=STAGE_FIRST,
        )
        assert "reconnect" in system.lower() or "3+" in system or \
               "acknowledge the gap" in system.lower()

    def test_stage_ending_short_warm(self):
        from brain.generator.reply import build_prompt_v6
        from brain.generator.intent_sensors import (
            INTENT_PLANNING, STAGE_ENDING,
        )
        system, _ = build_prompt_v6(
            "theek hai bye milte hain", "Priya", None, None, [],
            intent=INTENT_PLANNING, stage=STAGE_ENDING,
        )
        assert "winding down" in system.lower() or "warm closing" in system.lower()

    def test_stage_left_read(self):
        from brain.generator.reply import build_prompt_v6
        from brain.generator.intent_sensors import (
            INTENT_GENERAL, STAGE_LEFT_READ,
        )
        system, _ = build_prompt_v6(
            "aur batao", "Priya", None, None, [],
            intent=INTENT_GENERAL, stage=STAGE_LEFT_READ,
        )
        assert "unread" in system.lower() or "delay" in system.lower() or \
               "left on read" in system.lower() or "busy" in system.lower()


# ══════════════════════════════════════════════════════════════════════════════
# ReplyGenerator v6 API — generate() returns (list[str], dict)
# ══════════════════════════════════════════════════════════════════════════════

class _NoOpStore:
    """Store stub — empty corpus, so retrieval always returns nothing."""
    def count(self):                           return 0
    def retrieve_similar(self, *a, **kw):      return []
    def retrieve_similar_lang_aware(self, *a, **kw): return []


@pytest.fixture
def gen(tmp_path, fake_ollama):
    """ReplyGenerator wired to _NoOpStore + the canned LLM response."""
    from brain.generator.reply import ReplyGenerator
    return ReplyGenerator(_NoOpStore(), profiles_dir=tmp_path)


class TestReplyGeneratorV6NoOllama:
    """
    generate()'s v6 contract: (drafts, explainability).

    fake_ollama replaces ollama.generate, so these stay offline and fast while
    still asserting on real drafts — the old versions hit the live LLM and the
    draft assertions passed vacuously because they were never non-empty.
    """

    def test_generate_returns_tuple(self, gen):
        result = gen.generate("hello how are you?", "Priya", "TestUser")
        assert isinstance(result, tuple)
        assert len(result) == 2
        drafts, explain = result
        assert isinstance(drafts, list)
        assert isinstance(explain, dict)

    def test_drafts_are_non_empty_strings(self, gen):
        drafts, _ = gen.generate("hello how are you?", "Priya", "TestUser")
        assert len(drafts) > 0, "canned LLM response must survive the pipeline"
        assert drafts == _CANNED_DRAFTS
        for d in drafts:
            assert isinstance(d, str) and len(d.strip()) > 0

    def test_generate_legacy_returns_list(self, gen):
        drafts = gen.generate_legacy("hi", "Priya", "TestUser")
        assert isinstance(drafts, list)
        assert len(drafts) > 0
        assert drafts == _CANNED_DRAFTS

    def test_smart_silence_ok_returns_empty_drafts(self, gen, fake_ollama):
        drafts, explain = gen.generate("ok", "Priya", "TestUser")
        assert drafts == []
        assert explain.get("silence_skip") is True
        assert len(fake_ollama.calls) == 0, "silence must short-circuit before the LLM"

    def test_smart_silence_haan_no_drafts(self, gen, fake_ollama):
        drafts, explain = gen.generate("haan", "Priya", "TestUser")
        assert drafts == []
        assert explain["silence_skip"] is True
        assert len(fake_ollama.calls) == 0, "silence must short-circuit before the LLM"

    def test_explainability_has_keys(self, gen):
        _, explain = gen.generate("hello there!", "Priya", "TestUser")
        assert isinstance(explain, dict)
        for k in ("intent", "emotion", "language", "silence_skip"):
            assert k in explain

    def test_question_incoming_gets_question_intent(self, gen):
        drafts, explain = gen.generate("free tonight?", "Priya", "TestUser")
        assert explain["intent"] == "question"
        assert len(drafts) > 0

    def test_language_detection_in_explainability(self, gen):
        _, explain = gen.generate(
            "haan bhai ghar pahuch gaya", "Priya", "TestUser",
        )
        assert explain["language"] in ("english", "hinglish")

    def test_last_explainability(self, gen):
        gen.generate("heyy how are you?", "Priya", "TestUser")
        d = gen.last_explainability()
        assert isinstance(d, dict)
        assert len(d) > 0
        assert d == gen._last_explainability