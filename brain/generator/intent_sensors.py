"""
brain/generator/intent_sensors.py

Pure heuristics for intent, emotion, conversation stage, and language detection.

Design:
  All rule-based, zero external dependencies, zero API calls.
  Each sensor is a standalone callable — composable into the reply pipeline.

Features powered:
  3. Intent-aware reply generation
  4. Conversation stage awareness
  5. Emotional context
  6. Language-aware retrieval
  8. Smart silence
  7. Explainability builder
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# Intent detection
# ══════════════════════════════════════════════════════════════════════════════

INTENT_GREETING = "greeting"
INTENT_QUESTION = "question"
INTENT_PLANNING = "planning"
INTENT_TASK     = "task"
INTENT_EMOTIONAL = "emotional"
INTENT_HUMOR    = "humor"
INTENT_GENERAL  = "general"

GREETING_PATTERNS = re.compile(
    r"^(hi|hey|hello|hii+|heyy+|heyyy+|yo+|yoo+|sup|wassup|what's up|"
    r"howdy|gm|good morning|good evening|namaste|namaskar|salaam|"
    r"kese ho|kaise ho|kaise hai|kya haal|aur bata|aur batao)\b",
    re.IGNORECASE,
)

QUESTION_MARKERS = ["?", "kya", "kaise", "kab", "kitna", "kitne", "kahan",
                    "kyu", "kisne", "kisko", "what", "how", "when", "where",
                    "why", "who", "which", "tell me", "bata", "batao"]

PLANNING_KEYWORDS = [
    "free", "milte", "milna", "mil le", "aaja", "aa ja", "chal",
    "chale", "plan", "scene", "tonight", "aaj", "kal", "shaam",
    "evening", "weekend", "movie", "canteen", "coffee", "chai",
    "hangout", "hang out", "meet", "party", "chill",
]

TASK_KEYWORDS = [
    "assignment", "project", "deadline", "submit", "submit karna",
    "bhej", "bhej de", "send", "share", "forward", "mail", "file",
    "pdf", "notes", "proxy", "class", "lecture", "exam", "paper",
    "report", "review", "fix", "update", "change", "add", "remove",
]

EMOTIONAL_KEYWORDS = [
    "stress", "tension", "mood off", "sad", "darr", "dar", "worried",
    "scared", "upset", "depressed", "angry", "gussa", "annoyed",
    "frustrated", "miss you", "lonely", "hurt", "break up",
    "problem", "help", "mujhe help chahiye", "bura lag raha",
]

HUMOR_KEYWORDS = [
    "lol", "lmao", "lmfao", "haha", "hehe", "funny", "joke",
    "masti", "mazak", "😂", "🤣", "💀", "rofl",
]


def detect_intent(text: str) -> str:
    """
    Return the dominant intent of an incoming message.

    Priority: greeting > question > task > planning > emotional > humor > general
    """
    if not text or not text.strip():
        return INTENT_GENERAL

    text_l = text.lower().strip()

    if GREETING_PATTERNS.match(text_l) and len(text.split()) <= 5:
        return INTENT_GREETING

    if "?" in text:
        return INTENT_QUESTION
    if any(text_l.startswith(w) for w in QUESTION_MARKERS):
        return INTENT_QUESTION

    task_hits = sum(1 for kw in TASK_KEYWORDS if kw in text_l)
    if task_hits >= 2:
        return INTENT_TASK

    plan_hits = sum(1 for kw in PLANNING_KEYWORDS if kw in text_l)
    if plan_hits >= 2:
        return INTENT_PLANNING

    emotional_hits = sum(1 for kw in EMOTIONAL_KEYWORDS if kw in text_l)
    if emotional_hits >= 2:
        return INTENT_EMOTIONAL
    if emotional_hits >= 1 and len(text.split()) <= 6:
        return INTENT_EMOTIONAL

    humor_hits = sum(1 for kw in HUMOR_KEYWORDS if kw in text_l)
    if humor_hits >= 1 and len(text.split()) <= 8:
        return INTENT_HUMOR

    return INTENT_GENERAL


# ══════════════════════════════════════════════════════════════════════════════
# Emotion detection
# ══════════════════════════════════════════════════════════════════════════════

EMOTION_HAPPY   = "happy"
EMOTION_SAD     = "sad"
EMOTION_ANGRY   = "angry"
EMOTION_NEUTRAL = "neutral"
EMOTION_EXCITED = "excited"

HAPPY_WORDS = {
    "happy", "khush", "mast", "maza", "awesome", "great", "nice",
    "badhiya", "badhia", "sahi", "sahi hai", "best", "wow",
    "woohoo", "yay", "let's go", "lets go", "🎉", "🥳",
}

SAD_WORDS = {
    "sad", "dukhi", "dukh", "tension", "stress", "mood off",
    "bura", "bura lag raha", "alone", "lonely", "missing",
    "miss you", "😢", "😭", "💔",
}

ANGRY_WORDS = {
    "angry", "gussa", "annoyed", "frustrated", "hadd hai",
    "pagal hai kya", "stupid", "idiot", "wtf", "bakwas",
    "bakwaas", "chutiya", "😡", "🤬",
}

EXCITED_WORDS = {
    "excited", "let's go", "lets go", "finally", "yess", "yesss",
    "omg", "no way", "awesome", "amazing", "🔥", "💯", "🙌",
    "!!", "!!!",
}


def detect_emotion(text: str) -> str:
    """
    Return the dominant emotional tone of an incoming message.
    """
    if not text or not text.strip():
        return EMOTION_NEUTRAL

    text_l = text.lower()

    scores: dict[str, int] = {
        EMOTION_HAPPY:   0,
        EMOTION_SAD:     0,
        EMOTION_ANGRY:   0,
        EMOTION_EXCITED: 0,
    }

    for word in HAPPY_WORDS:
        if word in text_l:
            scores[EMOTION_HAPPY] += 1
    for word in SAD_WORDS:
        if word in text_l:
            scores[EMOTION_SAD] += 1
    for word in ANGRY_WORDS:
        if word in text_l:
            scores[EMOTION_ANGRY] += 1
    for word in EXCITED_WORDS:
        if word in text_l:
            scores[EMOTION_EXCITED] += 1

    exclamation_count = text.count("!") + text.count("!!")
    if exclamation_count >= 2 and scores[EMOTION_EXCITED] == 0:
        scores[EMOTION_EXCITED] = 1
    if "!!!" in text:
        scores[EMOTION_EXCITED] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return EMOTION_NEUTRAL
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Conversation stage detection
# ══════════════════════════════════════════════════════════════════════════════

STAGE_FIRST      = "first_message"
STAGE_MID        = "mid_conversation"
STAGE_LEFT_READ  = "left_on_read"
STAGE_ENDING     = "conversation_ending"
STAGE_UNKNOWN    = "unknown"


@dataclass
class ConversationContext:
    last_interaction_ts: str | None = None
    hours_since_last: float | None = None
    previous_incoming: str | None = None
    previous_my_reply: str | None = None
    unread_count: int = 0


def detect_stage(
    incoming: str,
    context: ConversationContext | None = None,
) -> str:
    """
    Detect where we are in a conversation.

    Rules:
      >72 hours since last interaction → first_message (or new thread)
      6+ hours gap                               → left_on_read
      Incoming contains bye/ok/goodnight         → ending
      Otherwise                                  → mid_conversation
    """
    if not context:
        return STAGE_UNKNOWN

    text_l = incoming.lower().strip()

    ending_markers = [
        "bye", "byee", "gn", "good night", "goodnight", "tc", "take care",
        "ok bye", "theek hai bye", "chal bye", "meeting", "milte hain",
        "so ja", "so jao", "sone ja", "kal baat karte",
    ]
    if any(text_l.endswith(m) or text_l.startswith(m) for m in ending_markers):
        return STAGE_ENDING if len(incoming.split()) <= 4 else STAGE_MID

    if context.hours_since_last is not None:
        if context.hours_since_last > 72:
            return STAGE_FIRST
        if context.hours_since_last > 6 and context.unread_count > 0:
            return STAGE_LEFT_READ

    return STAGE_MID


# ══════════════════════════════════════════════════════════════════════════════
# Language detection (Hindi / Hinglish / English)
# ══════════════════════════════════════════════════════════════════════════════

LANG_HINGLISH = "hinglish"
LANG_ENGLISH  = "english"

HINDI_FUNCTION_WORDS = {
    "hai", "hain", "ho", "tha", "thi", "the", "thi",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par",
    "ne", "bhi", "hi", "toh", "to", "bhi", "nahi", "nahin",
    "kar", "karna", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "hu", "hun", "hoon", "kya", "kyu", "kyun", "kaise", "kab",
    "kahan", "kitna", "kitne", "kisne", "kisko", "lekin",
    "aur", "ya", "fir", "phir", "ab", "abhi", "bas", "bus",
    "haan", "haa", "han", "na", "naa", "theek", "achha", "acha",
    "accha", "sahi", "galat", "pata", "malum", "maloom",
    "de", "do", "le", "lo", "aaya", "aayi", "aaye", "ja",
    "jao", "jaa", "aa", "aao", "chal", "chalo", "ruk", "ruko",
    "dekh", "dekho", "sun", "suno", "bol", "bolo", "bata", "batao",
    "kha", "khao", "pi", "piyo", "so", "soja", "uth", "utho",
    "karunga", "karega", "karegi", "karoge", "karoge",
    "aunga", "aaunga", "jaunga", "bhej", "bhej de", "bhej do",
    "mil", "milte", "milenge", "dekhunga", "batunga",
    "ghar", "kamra", "room", "class", "college", "office",
    "dost", "yaar", "bhai", "didi", "maa", "papa", "mom", "dad",
}


def detect_language(text: str) -> str:
    """
    Heuristic: if >= 20% of tokens match known Hinglish/Hindi function words → Hinglish.
    Otherwise English.

    This is used for language-aware RAG retrieval: prefer examples in
    the same language as the incoming message.
    """
    if not text or not text.strip():
        return LANG_ENGLISH

    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return LANG_ENGLISH

    hindi_hits = sum(1 for t in tokens if t in HINDI_FUNCTION_WORDS)
    ratio = hindi_hits / len(tokens)

    return LANG_HINGLISH if ratio >= 0.20 else LANG_ENGLISH


# ══════════════════════════════════════════════════════════════════════════════
# Smart silence — should we suggest a reply?
# ══════════════════════════════════════════════════════════════════════════════

SILENCE_PATTERNS = [
    "ok", "okay", "okkk", "okey", "k", "kk",
    "haan", "haa", "han", "haanji", "hmm", "hmmm",
    "theek hai", "theek", "thik hai", "thik",
    "achha", "acha", "accha", "oh", "oho",
    "done", "👍", "👌", "✅",
]


def should_skip_reply(incoming: str) -> bool:
    """
    Return True if the incoming message is too low-value to merit a reply draft.

    Skips:
      - Bare acknowledgements (ok, haan, theek hai)
      - Single emoji reactions
      - Very short acknowledgment patterns
    """
    stripped = incoming.strip().lower()
    if not stripped:
        return True

    if stripped in SILENCE_PATTERNS:
        return True
    if stripped.endswith("k") and len(stripped) <= 3:
        return True
    if re.fullmatch(r"[\U0001F600-\U0001FAFF\s]{1,3}", stripped):
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# Explainability record
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExplainabilityRecord:
    intent: str = INTENT_GENERAL
    emotion: str = EMOTION_NEUTRAL
    stage: str = STAGE_UNKNOWN
    language: str = LANG_ENGLISH
    relationship_type: str = "unknown"
    relationship_label: str = "friend"
    retrieved_examples_count: int = 0
    retrieved_example_previews: list[str] = field(default_factory=list)
    has_profile: bool = False
    has_relationship: bool = False
    silence_skip: bool = False
    silence_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "emotion": self.emotion,
            "stage": self.stage,
            "language": self.language,
            "relationship_type": self.relationship_type,
            "relationship_label": self.relationship_label,
            "retrieved_examples_count": self.retrieved_examples_count,
            "retrieved_example_previews": self.retrieved_example_previews,
            "has_profile": self.has_profile,
            "has_relationship": self.has_relationship,
            "silence_skip": self.silence_skip,
            "silence_reason": self.silence_reason,
        }

    def to_text(self) -> str:
        lines = ["── Why these drafts ──"]
        lines.append(f"Intent: {self.intent}")
        lines.append(f"Emotion detected: {self.emotion}")
        if self.stage != STAGE_UNKNOWN:
            lines.append(f"Conversation stage: {self.stage}")
        lines.append(f"Language: {self.language}")
        if self.has_profile:
            lines.append(f"Profile: loaded (style + emoji data)")
        else:
            lines.append("Profile: not available (generic tones used)")
        if self.has_relationship:
            lines.append(f"Relationship: {self.relationship_label}")
        else:
            lines.append("Relationship: unknown (generic tones used)")
        if self.retrieved_examples_count > 0:
            lines.append(
                f"Examples used: {self.retrieved_examples_count} "
                f"similar past messages"
            )
            for preview in self.retrieved_example_previews[:3]:
                lines.append(f"  → \"{preview}\"")
        else:
            lines.append("Examples used: none (no RAG matches found)")
        if self.silence_skip:
            lines.append(f"Skip reason: {self.silence_reason}")
        return "\n".join(lines)


def build_explainability(
    incoming: str,
    intent: str,
    emotion: str,
    stage: str,
    language: str,
    relationship_type: str,
    relationship_label: str,
    examples: list[dict],
    has_profile: bool,
    has_relationship: bool,
    silence_result: tuple[bool, str],
) -> ExplainabilityRecord:
    return ExplainabilityRecord(
        intent=intent,
        emotion=emotion,
        stage=stage,
        language=language,
        relationship_type=relationship_type,
        relationship_label=relationship_label,
        retrieved_examples_count=len(examples),
        retrieved_example_previews=[
            ex.get("content", "")[:80] for ex in examples[:5]
            if isinstance(ex.get("content", ""), str) and ex.get("content", "").strip()
        ],
        has_profile=has_profile,
        has_relationship=has_relationship,
        silence_skip=silence_result[0],
        silence_reason=silence_result[1],
    )