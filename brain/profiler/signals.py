"""
brain/profiler/signals.py

Conversation signal extraction for PersonaOS relationship classification.

This module is the foundation of context-aware relationship understanding.
It replaces raw word counting with structured signal extraction at multiple
levels: vocabulary, phrase patterns, behavioral patterns, topic clusters,
session dynamics, and emoji semantics.

Design principles:
  - Words are evidence, not primary classifiers
  - Phrases carry more meaning than isolated tokens
  - Conversation behavior reveals relationship structure
  - Topic distribution reveals relationship purpose
  - Compound signals override individual word signals

Architecture:
  extract_conversation_signals(user_msgs, all_msgs) → ConversationSignals
    ├── _extract_vocabulary()     normalized token matching
    ├── _extract_phrases()        multi-word pattern detection  
    ├── _extract_behaviors()      check-in / planning / emotional support
    ├── _extract_topics()         what the conversation is about
    ├── _extract_sessions()       structural dynamics
    ├── _extract_emoji()          semantic emoji grouping
    └── _detect_compounds()       joint signals (bhai+assignment → college)
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from brain.profiler.models import MessageRecord


# ══════════════════════════════════════════════════════════════════════════════
# VOCABULARY
# Organized by semantic purpose, not alphabetically.
# Words that belong to multiple sets are intentionally excluded from both
# and handled via compound signal detection instead.
# ══════════════════════════════════════════════════════════════════════════════

# ── English slang ─────────────────────────────────────────────────────────────
SLANG_WORDS = {
    "lol", "lmao", "lmfao", "omg", "wtf", "idk", "tbh", "ngl", "imo",
    "bruh", "bro", "dude", "yo", "nah", "yep", "yup", "oof", "rip",
    "bet", "fr", "frfr", "lowkey", "highkey", "slay", "vibe", "lit",
    "noice", "sheesh", "gg", "smh", "hmu", "nvm", "lmk", "cuz",
    "tho", "gonna", "wanna", "gotta", "tryna", "haha", "hehe",
    "u", "ur", "rn", "ik", "ikr",
}

# ── Hinglish core vocabulary ──────────────────────────────────────────────────
# "bhai" intentionally excluded — it's a compound signal, not a simple token.
# "ghar" excluded — too context-dependent.
HINDI_CORE = {
    # Address terms (non-ambiguous)
    "oye", "arre", "arrey", "yaar", "dost",
    # Acknowledgment
    "haan", "ji", "bilkul", "theek", "accha",
    # Negation
    "nahi", "mat",
    # Questions
    "kya", "kyun", "kaise", "kab", "kahan", "kaun", "kitna",
    # Connectors
    "toh", "aur", "ya", "lekin", "kyunki", "waise", "matlab",
    # Pronouns
    "mera", "tera", "uska", "humara", "apna", "mujhe",
    # Time
    "kal", "aaj", "abhi", "raat", "subah",
    # Common verbs (root forms)
    "kar", "karo", "jaa", "aana", "chal", "mil", "bhej",
    "bol", "sun", "dekh", "bata", "ruk",
    # Common conjugations
    "hai", "hain", "tha", "raha", "gaya", "hua",
    # Emotions/adjectives
    "khush", "pareshaan", "tension", "sahi", "galat", "mast",
    "zabardast", "bekar", "bakwas", "pagal",
    # Filler words
    "bohot", "thoda", "zyada", "bas", "toh",
}

# ── Hinglish abbreviation and phonetic variants ────────────────────────────────
# These map to their canonical HINDI_CORE equivalents after normalisation.
HINGLISH_ABBREVS: dict[str, str] = {
    # Acknowledgment short forms
    "hn": "haan",   "hnn": "haan",  "han": "haan",  "ha": "haan",
    "haa": "haan",  "hnji": "haan", "jii": "haan",
    # Negation short forms
    "nhi": "nahi",  "nahin": "nahi", "nahh": "nahi", "naa": "nahi",
    "naah": "nahi",
    # Agreement short forms
    "k": "okay",    "kk": "okay",   "okk": "okay",  "okie": "okay",
    "okies": "okay", "okayy": "okay", "okaaay": "okay",
    # Yaar short forms
    "yr": "yaar",   "yar": "yaar",
    # Bohot short forms
    "bhot": "bohot", "boht": "bohot", "bahot": "bohot",
    # Verb short forms
    "kr": "kar",    "krna": "karna", "krdia": "kar",  "krdeta": "kar",
    "krdunga": "kar", "kro": "karo",
    # Raha short forms
    "rha": "raha",  "rhaa": "raha", "rhe": "rahe",  "rhi": "rahi",
    # Aaja short forms
    "aaja": "aana", "aja": "aana",  "aajaa": "aana",
    # Movement
    "chl": "chal",
    # Spelling variants
    "thik": "theek", "theak": "theek",
    "acha": "accha", "achha": "accha",
    # Milte short forms
    "milte": "milna", "milo": "milna",
}

# ── College/campus-specific vocabulary ────────────────────────────────────────
# These are strong signals for academic_peer OR college_close_friend
# depending on compound context.
COLLEGE_ACADEMIC = {
    # Exam/assessment
    "midsem", "endsem", "cgpa", "sgpa", "backlog", "back",
    "attendance", "proxy", "marks", "grade", "result",
    # Course
    "assignment", "submission", "deadline", "notes", "lab",
    "practical", "viva", "project", "presentation", "ppt",
    # People
    "professor", "prof", "faculty", "hod", "ta", "batch",
    # Platforms
    "newtonschool", "classroom", "lms", "playground",
    # General
    "semester", "college", "university", "roll", "enrollment",
}

COLLEGE_SOCIAL = {
    # Places
    "canteen", "hostel", "mess", "campus", "library", "auditorium",
    # Events
    "hackathon", "ecell", "fest", "techfest", "culturals", "club",
    # Activities
    "pitch", "startup", "ideathon", "workshop",
    # Career
    "intern", "internship", "placement", "oa", "interview",
    "resume", "linkedin", "referral",
}

# ── Family terms — strong (unambiguous) ───────────────────────────────────────
# "bhai" and "ghar" removed — handled by compound detection
FAMILY_STRONG = {
    "maa", "mom", "mum", "mama",
    "dad", "papa", "paa",
    "didi", "bhaiya",
    "sis", "sister", "brother",
    "uncle", "aunty", "auntie",
    "nana", "nani", "dada", "dadi",
    "grandma", "grandpa", "gharwale",
}

# ── Nicknames and affectionate terms ──────────────────────────────────────────
# Presence signals CLOSE relationship (not necessarily romantic).
# These are stronger friendship signals than generic slang.
NICKNAME_TERMS = {
    "motu", "pagal", "gadha", "chomu", "champ", "bhidu",
    "yaar", "dost", "bhai",    # casual address — different from family "bhai"
    "jaan", "jaanu",           # can be friend OR romantic
    "paji", "paaji",
}

# ── Romantic/affection terms ──────────────────────────────────────────────────
ROMANTIC_TERMS = {
    "love", "babe", "baby", "darling", "sweetheart", "honey",
    "miss you", "missyou", "luv", "jaanu",
}

# ── Formal register markers ────────────────────────────────────────────────────
FORMAL_GREETINGS = re.compile(
    r"^\s*(dear\s+sir|dear\s+ma|dear\s+mr|dear\s+ms|hello\s+sir|"
    r"respected|to\s+whom|i\s+hope\s+you\s+are\s+doing\s+well|"
    r"i\s+am\s+writing\s+to)",
    re.IGNORECASE,
)

WORK_KEYWORDS = {
    "meeting", "client", "office", "agenda", "deliverable", "review",
    "feedback", "sprint", "standup", "budget", "invoice", "contract",
    "proposal", "quarterly", "kpi", "milestone", "manager", "hr",
    "onboarding",
}


# ══════════════════════════════════════════════════════════════════════════════
# PHRASE PATTERNS
# Multi-word sequences that carry meaning as a unit.
# Matched via substring against normalised message text.
# A phrase match is worth MORE than the same words counted separately.
# ══════════════════════════════════════════════════════════════════════════════

# ── Planning / meetup phrases ─────────────────────────────────────────────────
PLANNING_PHRASES = [
    # Meetup invitations
    "chal milte", "milte hain", "mil lete", "kal milte", "aaj milte",
    "aaja", "aa ja", "bahar chale", "bahar chalte", "canteen chale",
    "canteen chalte", "canteen?", "chill karte",
    # Plan inquiries
    "scene kya hai", "kya scene hai", "scene?", "kya plan hai",
    "plan kya hai", "aaj ka plan", "kal ka plan", "kya kar raha",
    "kya chal raha", "chal kya raha",
    # Social invitations
    "aajao", "aa jao", "chal aata", "chal aati", "let's go",
    "movie chalte", "khana chalte",
]

# ── Check-in phrases ──────────────────────────────────────────────────────────
# Asking about daily routines — signals CARE about the other person's daily life.
# Frequency of these matters more than single occurrence.
CHECKIN_PHRASES = [
    # Food check-in
    "khana khaya", "khaana khaya", "khaya", "khana kha",
    # Sleep/wake
    "so gaya", "so gayi", "uth gaya", "uth gayi", "uth gye",
    "sone ja", "so ja",
    # Arrival/location
    "ghar pahuch", "pahuch gaya", "pahunch gaya", "ghar aa gaya",
    "kahan hai", "kahan ho", "kab aoge", "kab aaoge",
    # General wellbeing
    "kya kar raha hai", "theek hai", "sab theek",
]

# ── Emotional support phrases ─────────────────────────────────────────────────
# Expressing or receiving emotional distress. Strongest trust signal.
EMOTIONAL_PHRASES = [
    # Distress
    "tension ho rahi", "tension mat le", "tension mat",
    "stress ho raha", "bahut stress", "itna stress",
    "mood off hai", "mood nahi", "mood kharab",
    "kuch ho gaya", "kuch hua kya", "sab theek hai",
    # Seeking support
    "yaar sun", "bhai sun", "samajh nahi aa raha",
    "kya karu", "kya karun", "advice de",
    # Offering support
    "bata mujhe", "teri madad", "main hun", "main hoon",
    "chillax yaar", "tension mat", "sab theek hoga",
    "chill kar",
]

# ── Academic task phrases ─────────────────────────────────────────────────────
# Collaboration on coursework — not just talking about academics
ACADEMIC_TASK_PHRASES = [
    "assignment bhej", "notes bhej", "pdf bhej", "link bhej",
    "share kar", "forward kar", "proxy laga", "proxy lagwa",
    "attendance laga", "attendance lagwa",
    "solution bhej", "code bhej", "answer bhej",
    "deadline hai", "submit karna hai", "submission aaj",
    "midsem hai", "endsem hai", "test hai",
]


# ══════════════════════════════════════════════════════════════════════════════
# EMOJI SEMANTIC GROUPS
# Emojis are grouped by emotional register, not just counted.
# A chat heavy in humor emojis signals very different relationship than
# one heavy in affection emojis, even if total emoji count is the same.
# ══════════════════════════════════════════════════════════════════════════════

EMOJI_HUMOR = {
    "😂", "🤣", "💀", "😭", "😹", "🫠", "💅", "🙃", "😵", "🤪",
    "😆", "😝", "🤡", "👀", "💁",
}

EMOJI_AFFECTION = {
    "❤️", "🥰", "😘", "😍", "💕", "💗", "💖", "💓", "🫶", "💞",
    "🤍", "💝", "💘", "😻",
}

EMOJI_ENERGY = {
    "🔥", "💯", "🙌", "🤩", "🥳", "✨", "🚀", "⚡", "🎉", "🎊",
    "💪", "👊",
}

EMOJI_SUPPORT = {
    "🤗", "🫂", "❤️‍🩹", "🙏", "👍", "✅",
}


# ══════════════════════════════════════════════════════════════════════════════
# COMPOUND SIGNAL PATTERNS
# These detect combinations within a SINGLE MESSAGE that produce a
# more specific classification than either word alone would.
# This solves the "bhai assignment bhej de" → college_friend problem.
# ══════════════════════════════════════════════════════════════════════════════

def _is_college_friend_compound(text: str) -> bool:
    """
    Detect: casual address + academic task = college close friend.
    "bhai assignment bhej de" → NOT split into family + academic.
    "yaar proxy laga" → close college friend.
    """
    text_l = text.lower()
    has_casual_address = any(w in text_l.split() for w in {"bhai", "yaar", "bro", "dost", "oye"})
    has_academic_task  = any(phrase in text_l for phrase in ACADEMIC_TASK_PHRASES) or \
                         any(w in text_l.split() for w in COLLEGE_ACADEMIC)
    return has_casual_address and has_academic_task


def _is_family_compound(text: str) -> bool:
    """
    Detect: unambiguous family term + logistics = family.
    "maa ghar kab aa rahi hai" → family (maa is unambiguous).
    "bhai ghar kab aa raha" → NOT family by default (bhai=casual address).
    """
    text_l = text.lower()
    has_strong_family = any(w in text_l.split() for w in FAMILY_STRONG)
    # "bhai" alone is NOT a strong family term
    if not has_strong_family:
        return False
    # Also require a logistics/check-in phrase to confirm family context
    # (parent/sibling asking about daily routine = strong family signal)
    has_logistics = any(phrase in text_l for phrase in CHECKIN_PHRASES)
    return has_logistics


def _is_planning_compound(text: str) -> bool:
    """Any planning phrase in the message."""
    text_l = text.lower()
    return any(phrase in text_l for phrase in PLANNING_PHRASES)


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION SIGNALS DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationSignals:
    """
    Structured output from conversation signal extraction.

    This is the interface between signal extraction (signals.py)
    and relationship scoring (classifier.py).

    Design: classifiers read from this object, never from raw messages.
    """

    # ── Vocabulary signals ─────────────────────────────────────────────────────
    slang_density:       float        = 0.0
    hindi_density:       float        = 0.0
    slang_top_words:     list[str]    = field(default_factory=list)
    hindi_top_words:     list[str]    = field(default_factory=list)
    college_academic_hits: int        = 0
    college_social_hits:   int        = 0
    work_hits:           int          = 0
    family_strong_hits:  int          = 0
    romantic_hits:       int          = 0
    nickname_hits:       int          = 0
    formal_count:        int          = 0

    # ── Phrase signals ────────────────────────────────────────────────────────
    planning_phrase_count:    int     = 0
    checkin_phrase_count:     int     = 0
    emotional_phrase_count:   int     = 0
    academic_task_phrase_count: int   = 0

    # ── Compound signals ──────────────────────────────────────────────────────
    college_friend_compound_count: int = 0
    family_compound_count:         int = 0

    # ── Behavioral pattern flags ──────────────────────────────────────────────
    # Set to True when pattern appears enough to be meaningful
    has_planning_behavior:         bool = False   # milte, chal, canteen?, aaja
    has_checkin_behavior:          bool = False   # khana khaya, so gaya
    has_emotional_support:         bool = False   # tension, stress, mood off
    has_academic_collaboration:    bool = False   # bhej, proxy, notes exchange

    # ── Topic distribution ────────────────────────────────────────────────────
    # What fraction of the conversation is about each topic cluster
    topic_academic:     float = 0.0   # coursework, exams, assignments
    topic_social:       float = 0.0   # plans, meetups, casual life
    topic_career:       float = 0.0   # internship, placement, hackathon
    topic_emotional:    float = 0.0   # feelings, support, personal
    topic_logistics:    float = 0.0   # where are you, are you home, daily
    topic_professional: float = 0.0   # work, meetings, formal
    dominant_topic:     str   = "general"

    # ── Session dynamics ──────────────────────────────────────────────────────
    session_count:        int   = 0
    avg_session_depth:    float = 0.0
    deep_session_rate:    float = 0.0   # % sessions with 10+ messages
    sessions_per_week:    float = 0.0
    initiation_rate:      float = 0.0   # % sessions user starts
    user_balance:         float = 0.5   # user's share of all messages
    avg_message_length:   float = 0.0
    late_night_rate:      float = 0.0
    emoji_rate:           float = 0.0
    media_rate:           float = 0.0

    # ── Emoji semantics ────────────────────────────────────────────────────────
    humor_emoji_rate:    float = 0.0   # 😂🤣💀 — casual/close friend
    affection_emoji_rate: float = 0.0  # ❤️🥰😘 — close/romantic
    energy_emoji_rate:   float = 0.0   # 🔥💯🙌 — enthusiastic/casual

    # ── Evidence log ──────────────────────────────────────────────────────────
    # Human-readable signals for display in CLI `signals` command
    evidence: list[str] = field(default_factory=list)

    def add_evidence(self, text: str) -> None:
        if text not in self.evidence:
            self.evidence.append(text)


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(word: str) -> str:
    """Normalize a single token: lowercase, strip punctuation, apply abbrevs."""
    w = word.lower().strip(".,!?\"';:()[]🏻🏼🏽🏾🏿")
    return HINGLISH_ABBREVS.get(w, w)


def _tokenize_normalized(text: str) -> list[str]:
    """Tokenize message text with full normalization applied."""
    return [_normalize(w) for w in text.split() if w.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_conversation_signals(
    user_msgs: list[MessageRecord],
    all_msgs: list[MessageRecord],
) -> ConversationSignals:
    """
    Analyze a conversation and return structured ConversationSignals.

    This is the single entry point for all signal extraction.
    The classifier only reads from the returned ConversationSignals object
    — it never touches raw messages directly.

    Args:
        user_msgs: Messages sent BY the user (the person being profiled)
        all_msgs:  All messages in this conversation (both sides)
    """
    signals = ConversationSignals()
    text_msgs = [m for m in user_msgs if m.content.strip()]

    if not text_msgs:
        return signals

    _extract_vocabulary(text_msgs, signals)
    _extract_phrases(text_msgs, signals)
    _detect_compound_signals(text_msgs, signals)
    _extract_behaviors(signals)
    _extract_topics(text_msgs, signals)
    _extract_sessions(user_msgs, all_msgs, signals)
    _extract_emoji(text_msgs, signals)

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_vocabulary(text_msgs: list[MessageRecord], signals: ConversationSignals) -> None:
    """Token-level vocabulary matching with full normalization."""
    slang_counter:    Counter = Counter()
    hindi_counter:    Counter = Counter()
    academic_counter: Counter = Counter()
    social_counter:   Counter = Counter()
    work_counter:     Counter = Counter()
    family_counter:   Counter = Counter()
    romantic_counter: Counter = Counter()
    nickname_counter: Counter = Counter()

    total_words = 0
    formal_count = 0

    for m in text_msgs:
        if FORMAL_GREETINGS.match(m.content):
            formal_count += 1

        tokens = _tokenize_normalized(m.content)
        total_words += len(tokens)

        for w in tokens:
            if w in SLANG_WORDS:       slang_counter[w]    += 1
            if w in HINDI_CORE:        hindi_counter[w]    += 1
            if w in COLLEGE_ACADEMIC:  academic_counter[w] += 1
            if w in COLLEGE_SOCIAL:    social_counter[w]   += 1
            if w in WORK_KEYWORDS:     work_counter[w]     += 1
            if w in FAMILY_STRONG:     family_counter[w]   += 1
            if w in ROMANTIC_TERMS:    romantic_counter[w] += 1
            if w in NICKNAME_TERMS:    nickname_counter[w] += 1

    total_words = total_words or 1

    signals.slang_density   = sum(slang_counter.values()) / total_words
    signals.hindi_density   = sum(hindi_counter.values()) / total_words
    signals.slang_top_words = _fmt_counter(slang_counter)
    signals.hindi_top_words = _fmt_counter(hindi_counter)
    signals.college_academic_hits = len(academic_counter)
    signals.college_social_hits   = len(social_counter)
    signals.work_hits             = len(work_counter)
    signals.family_strong_hits    = sum(family_counter.values())
    signals.romantic_hits         = sum(romantic_counter.values())
    signals.nickname_hits         = sum(nickname_counter.values())
    signals.formal_count          = formal_count

    signals.avg_message_length = (
        sum(m.char_count for m in text_msgs) / len(text_msgs)
    )

    total = len(text_msgs)
    signals.late_night_rate = sum(
        1 for m in text_msgs
        if m.timestamp.hour >= 23 or m.timestamp.hour <= 4
    ) / total if total else 0.0

    signals.emoji_rate = sum(1 for m in text_msgs if m.has_emoji) / total if total else 0.0
    signals.media_rate = sum(1 for m in text_msgs if m.has_media) / total if total else 0.0


def _extract_phrases(text_msgs: list[MessageRecord], signals: ConversationSignals) -> None:
    """
    Multi-word phrase pattern matching.

    Phrases are matched as substrings of the normalized message text.
    A phrase match is intentionally worth MORE than the sum of its words
    because the phrase carries a specific communicative function.
    """
    for m in text_msgs:
        text_l = m.content.lower()

        # Check phrase lists
        for phrase in PLANNING_PHRASES:
            if phrase in text_l:
                signals.planning_phrase_count += 1
                break   # count message once even if multiple phrases match

        for phrase in CHECKIN_PHRASES:
            if phrase in text_l:
                signals.checkin_phrase_count += 1
                break

        for phrase in EMOTIONAL_PHRASES:
            if phrase in text_l:
                signals.emotional_phrase_count += 1
                break

        for phrase in ACADEMIC_TASK_PHRASES:
            if phrase in text_l:
                signals.academic_task_phrase_count += 1
                break


def _detect_compound_signals(text_msgs: list[MessageRecord], signals: ConversationSignals) -> None:
    """
    Detect compound signals within individual messages.

    These resolve the ambiguity that arises from counting words separately.
    "bhai assignment bhej de" → college_friend_compound (not family + academic).
    """
    for m in text_msgs:
        if _is_college_friend_compound(m.content):
            signals.college_friend_compound_count += 1
        if _is_family_compound(m.content):
            signals.family_compound_count += 1


def _extract_behaviors(signals: ConversationSignals) -> None:
    """
    Set behavioral flags based on phrase counts.

    Thresholds are intentionally low — we care about presence of a
    behavior pattern, not its exact frequency.
    """
    total_msgs = max(signals.planning_phrase_count +
                     signals.checkin_phrase_count +
                     signals.emotional_phrase_count, 1)

    signals.has_planning_behavior      = signals.planning_phrase_count >= 3
    signals.has_checkin_behavior       = signals.checkin_phrase_count >= 3
    signals.has_emotional_support      = signals.emotional_phrase_count >= 2
    signals.has_academic_collaboration = signals.academic_task_phrase_count >= 3


def _extract_topics(text_msgs: list[MessageRecord], signals: ConversationSignals) -> None:
    """
    Topic cluster scoring — what is this conversation about?

    Each topic cluster has a vocabulary set. For each message, we count
    how many cluster words appear. The topic with the most representation
    across the conversation is the dominant topic.

    This captures relationship purpose, not just relationship tone.
    Academic_peer chats have high academic topic density.
    Close_friend chats have high social topic density.
    Family chats have high logistics topic density.
    """
    topic_scores: dict[str, int] = {
        "academic":     0,
        "social":       0,
        "career":       0,
        "emotional":    0,
        "logistics":    0,
        "professional": 0,
    }

    # Topic vocabularies (inline — small enough to not need separate constants)
    topic_vocab = {
        "academic":     COLLEGE_ACADEMIC,
        "social":       COLLEGE_SOCIAL | {"chill", "plan", "movie", "party", "fun"},
        "career":       {"intern", "internship", "placement", "oa", "hackathon",
                         "ecell", "startup", "pitch", "resume", "linkedin"},
        "emotional":    {"tension", "stress", "pareshaan", "mood", "feel",
                         "worried", "anxious", "depressed", "happy", "scared",
                         "emotional", "feelings", "khush", "dukhi"},
        "logistics":    {"ghar", "khana", "so", "uth", "pahuch", "kahan",
                         "aa raha", "nikal", "late", "waiting", "bata"},
        "professional": WORK_KEYWORDS,
    }

    total_matches = 0
    for m in text_msgs:
        tokens = set(_tokenize_normalized(m.content))
        for topic, vocab in topic_vocab.items():
            hits = len(tokens & vocab)
            topic_scores[topic] += hits
            total_matches += hits

    total_matches = total_matches or 1

    signals.topic_academic     = topic_scores["academic"]     / total_matches
    signals.topic_social       = topic_scores["social"]       / total_matches
    signals.topic_career       = topic_scores["career"]       / total_matches
    signals.topic_emotional    = topic_scores["emotional"]    / total_matches
    signals.topic_logistics    = topic_scores["logistics"]    / total_matches
    signals.topic_professional = topic_scores["professional"] / total_matches

    signals.dominant_topic = max(topic_scores, key=topic_scores.get)


def _extract_sessions(
    user_msgs: list[MessageRecord],
    all_msgs: list[MessageRecord],
    signals: ConversationSignals,
) -> None:
    """Session-level conversation dynamics."""
    if len(all_msgs) < 3:
        return

    SESSION_GAP_HOURS = 4
    sorted_msgs = sorted(all_msgs, key=lambda m: m.timestamp)

    # Segment into sessions
    sessions: list[list[MessageRecord]] = []
    current: list[MessageRecord] = [sorted_msgs[0]]

    for msg in sorted_msgs[1:]:
        gap = (msg.timestamp - current[-1].timestamp).total_seconds() / 3600
        if gap > SESSION_GAP_HOURS:
            sessions.append(current)
            current = [msg]
        else:
            current.append(msg)
    sessions.append(current)

    user_senders = {m.sender for m in user_msgs}
    user_name = next(iter(user_senders), "")

    count = len(sessions)
    signals.session_count = count
    signals.avg_session_depth = sum(len(s) for s in sessions) / count if count else 0.0
    signals.deep_session_rate = sum(1 for s in sessions if len(s) >= 10) / count if count else 0.0
    signals.initiation_rate = (
        sum(1 for s in sessions if s[0].sender == user_name) / count if count else 0.0
    )
    signals.user_balance = len(user_msgs) / len(all_msgs) if all_msgs else 0.5

    if len(sorted_msgs) > 1:
        total_days = max((sorted_msgs[-1].timestamp - sorted_msgs[0].timestamp).days, 1)
        signals.sessions_per_week = count / (total_days / 7)
    else:
        signals.sessions_per_week = 0.0


def _extract_emoji(text_msgs: list[MessageRecord], signals: ConversationSignals) -> None:
    """Emoji semantic grouping — humor vs affection vs energy."""
    humor_count    = 0
    affection_count = 0
    energy_count   = 0
    total          = len(text_msgs) or 1

    for m in text_msgs:
        if not m.has_emoji:
            continue
        text = m.content
        if any(e in text for e in EMOJI_HUMOR):
            humor_count += 1
        if any(e in text for e in EMOJI_AFFECTION):
            affection_count += 1
        if any(e in text for e in EMOJI_ENERGY):
            energy_count += 1

    signals.humor_emoji_rate    = humor_count    / total
    signals.affection_emoji_rate = affection_count / total
    signals.energy_emoji_rate   = energy_count   / total


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_counter(counter: Counter, limit: int = 4) -> list[str]:
    return [
        f"{w} ({c}×)"
        for w, c in counter.most_common(limit)
        if c >= 2
    ]