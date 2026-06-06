"""
brain/profiler/classifier.py

Infers the relationship type between a user and a contact
from message data alone — no LLM required.

How it works:
    1. Extract a set of numeric/boolean features from the messages
    2. Score those features against each relationship category
    3. Return the highest-scoring category with confidence + evidence

Design principles:
    - Heuristic only (fast, local, no API calls)
    - Explainable (every score has a human-readable signal)
    - Tunable (all thresholds are named constants at the top)
    - Conservative (defaults to "acquaintance" on low signal)

Relationship categories:
    close_friend     High slang, casual, late-night, personal, Hinglish
    academic_peer    Shared coursework, assignments, academic links
    family           Family terms, practical logistics messages
    colleague        Professional tone, work projects, formal language
    romantic_interest Terms of endearment, very personal, high emoji
    acquaintance     Low message count, surface-level, minimal slang
    unknown          Insufficient data to classify (< MIN_MESSAGES)
"""
import re
from datetime import time as dtime
from brain.profiler.models import MessageRecord


# ── Tunable thresholds ─────────────────────────────────────────────────────────

MIN_MESSAGES = 10             # minimum messages needed for any classification
LATE_NIGHT_START = 23         # hour (24h) after which a message is "late night"
LATE_NIGHT_END = 4            # hour (24h) before which a message is still "late night"
HIGH_SLANG_THRESHOLD = 0.04   # slang words / total words ratio
HIGH_EMOJI_THRESHOLD = 0.30   # messages with emoji / total messages
HIGH_MEDIA_THRESHOLD = 0.35   # media messages / total messages
LOW_AVG_LENGTH = 35           # chars — below this = terse/casual style
HIGH_AVG_LENGTH = 120         # chars — above this = formal/detailed style
HINGLISH_THRESHOLD = 0.06     # Hindi words / total words
CONFIDENCE_FLOOR = 0.39       # minimum score to report a non-unknown type


# ── Vocabulary sets ────────────────────────────────────────────────────────────

SLANG_WORDS = {
    # English slang
    "lol", "lmao", "lmfao", "omg", "wtf", "idk", "tbh", "ngl", "imo",
    "bruh", "bro", "dude", "yo", "nah", "yep", "yup", "oof", "rip",
    "bet", "fr", "frfr", "lowkey", "highkey", "slay", "vibe", "lit",
    "noice", "sheesh", "gg", "smh", "hmu", "nvm", "lmk", "bc", "cuz",
    "tho", "gonna", "wanna", "gotta", "tryna", "haha", "hehe",
    # Common informal shortened forms
    "u", "ur", "r", "rn", "ik", "ikr",
}

# Romanised Hindi/Hinglish words — signals Indian peer context
HINDI_WORDS = {
    "yaar", "bhai", "oye", "arre", "haan", "nahi", "nhi", "kya",
    "toh", "aur", "bhi", "abhi", "thoda", "zyada", "bohot", "bahut",
    "accha", "theek", "sahi", "bilkul", "matlab", "agar", "phir",
    "mein", "mai", "tera", "mera", "uska", "humara", "hnn", "hai",
    "tha", "the", "thi", "rha", "rhe", "rhaa", "kr", "kar", "karo",
    "bhej", "bhejio", "dekh", "sun", "bol", "bata", "aa", "jaa",
    "dekha", "suna", "bola", "hua", "hui", "hue",
}

# Academic / coursework keywords
ACADEMIC_KEYWORDS = {
    "assignment", "homework", "project", "deadline", "submission",
    "submit", "professor", "prof", "lecture", "class", "course",
    "exam", "test", "quiz", "marks", "grade", "semester", "pdf",
    "notes", "study", "lab", "presentation", "ppt", "report",
    "affinity", "empathy", "scamper", "brainstorm", "research",
    "college", "university", "batch", "roll", "enrollment",
    # Newton School / online course references
    "newtonschool", "playground", "code",
}

# Work / professional keywords
WORK_KEYWORDS = {
    "meeting", "client", "office", "email", "agenda", "deliverable",
    "review", "feedback", "sprint", "standup", "budget", "invoice",
    "contract", "proposal", "quarterly", "kpi", "milestone", "manager",
    "team", "colleague", "hr", "onboarding", "interview",
}

# Family terms
FAMILY_TERMS = {
    "mom", "mum", "maa", "mama", "dad", "papa", "paa", "bhai",
    "didi", "bhaiya", "sis", "sister", "brother", "uncle", "aunty",
    "auntie", "nana", "nani", "dada", "dadi", "grandma", "grandpa",
    "family", "ghar", "home", "gharpe", "gharwale",
}

# Terms of endearment / romantic signals
ROMANTIC_TERMS = {
    "love", "babe", "baby", "darling", "sweetheart", "honey",
    "miss you", "missyou", "luv", "❤", "💕", "💗", "💓", "😘",
    "😍", "🥰", "cute", "beautiful", "handsome",
}

# Formal greeting patterns
FORMAL_GREETINGS = re.compile(
    r"^\s*(dear\s+sir|dear\s+ma|dear\s+mr|dear\s+ms|hello\s+sir|"
    r"respected|to\s+whom|i\s+hope\s+you\s+are\s+doing\s+well|"
    r"i\s+am\s+writing\s+to)",
    re.IGNORECASE,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_relationship(
    user_messages: list[MessageRecord],
    all_messages: list[MessageRecord],
) -> tuple[str, float, list[str]]:
    """
    Infer the relationship type from message data.

    Args:
        user_messages: Messages sent BY the user to this contact
        all_messages:  All messages in this chat (both sides)

    Returns:
        (relationship_type, confidence, signals)
        - relationship_type: one of RELATIONSHIP_TYPES
        - confidence: 0.0–1.0
        - signals: human-readable list of evidence
    """
    if len(user_messages) < MIN_MESSAGES:
        return "unknown", 0.0, ["insufficient message count for classification"]

    features = _extract_features(user_messages, all_messages)
    scores, signals = _score_categories(features)

    if not scores:
        return "unknown", 0.0, ["no strong signals detected"]

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Normalise score to 0–1 confidence (cap raw score at 10)
    confidence = round(min(best_score / 10.0, 1.0), 2)

    if confidence < CONFIDENCE_FLOOR:
        return "acquaintance", confidence, signals.get(best_type, [])

    return best_type, confidence, signals.get(best_type, [])


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_features(
    user_msgs: list[MessageRecord],
    all_msgs: list[MessageRecord],
) -> dict:
    """Compute numeric features from the message corpus."""

    total = len(user_msgs)
    text_msgs = [m for m in user_msgs if m.content.strip()]
    all_words: list[str] = []
    for m in text_msgs:
        all_words.extend(m.content.lower().split())

    total_words = len(all_words) or 1

    # ── Rates ──────────────────────────────────────────────────────────────────
    emoji_rate = sum(1 for m in user_msgs if m.has_emoji) / total
    media_rate = sum(1 for m in user_msgs if m.has_media) / total
    late_night_rate = sum(
        1 for m in user_msgs
        if m.timestamp.hour >= LATE_NIGHT_START or m.timestamp.hour <= LATE_NIGHT_END
    ) / total
    avg_length = (
        sum(m.char_count for m in text_msgs) / len(text_msgs)
        if text_msgs else 0
    )

    # ── Vocabulary analysis ────────────────────────────────────────────────────
    word_set = set(all_words)
    slang_count = sum(1 for w in all_words if w.strip(".,!?\"'") in SLANG_WORDS)
    slang_density = slang_count / total_words

    hindi_count = sum(1 for w in all_words if w.strip(".,!?\"'") in HINDI_WORDS)
    hindi_density = hindi_count / total_words

    academic_hits = sum(
        1 for w in word_set
        if any(kw in w for kw in ACADEMIC_KEYWORDS)
    )
    work_hits = sum(
        1 for w in word_set
        if any(kw in w for kw in WORK_KEYWORDS)
    )
    family_hits = sum(
        1 for w in word_set
        if any(ft in w for ft in FAMILY_TERMS)
    )
    romantic_hits = sum(
        1 for w in all_words
        if w.strip(".,!?\"'") in ROMANTIC_TERMS
    ) + sum(
        1 for m in user_msgs
        if any(term in m.content.lower() for term in ROMANTIC_TERMS)
    )
    formal_count = sum(
        1 for m in text_msgs
        if FORMAL_GREETINGS.match(m.content)
    )

    # ── URL / link sharing ─────────────────────────────────────────────────────
    url_msgs = sum(1 for m in text_msgs if "http" in m.content.lower())
    academic_url_msgs = sum(
        1 for m in text_msgs
        if any(kw in m.content.lower() for kw in ["newtonschool", "classroom", "drive.google", "forms.gle"])
    )

    return {
        "total": total,
        "emoji_rate": emoji_rate,
        "media_rate": media_rate,
        "late_night_rate": late_night_rate,
        "avg_length": avg_length,
        "slang_density": slang_density,
        "hindi_density": hindi_density,
        "academic_hits": academic_hits,
        "work_hits": work_hits,
        "family_hits": family_hits,
        "romantic_hits": romantic_hits,
        "formal_count": formal_count,
        "url_msgs": url_msgs,
        "academic_url_msgs": academic_url_msgs,
        "word_set": word_set,
    }


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_categories(f: dict) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    Score each relationship category against the extracted features.
    Returns (scores dict, signals dict) — one entry per category.
    """
    scores: dict[str, float] = {}
    signals: dict[str, list[str]] = {}

    # ── close_friend ──────────────────────────────────────────────────────────
    cf_score = 0.0
    cf_signals = []
    if f["slang_density"] > HIGH_SLANG_THRESHOLD:
        cf_score += 2.5
        cf_signals.append(f"heavy slang usage ({f['slang_density']:.1%} of words)")
    if f["hindi_density"] > HINGLISH_THRESHOLD:
        cf_score += 2.0
        cf_signals.append("Hindi-English code-switching (Hinglish)")
    if f["late_night_rate"] > 0.20:
        cf_score += 2.0
        cf_signals.append(f"frequently messages late at night ({f['late_night_rate']:.0%} of messages)")
    if f["emoji_rate"] > HIGH_EMOJI_THRESHOLD:
        cf_score += 1.0
        cf_signals.append(f"high emoji usage ({f['emoji_rate']:.0%} of messages)")
    if f["avg_length"] < LOW_AVG_LENGTH:
        cf_score += 1.0
        cf_signals.append("short, casual message style")
    if f["formal_count"] == 0 and f["total"] > 20:
        cf_score += 0.5
        cf_signals.append("no formal greetings detected")
    scores["close_friend"] = cf_score
    signals["close_friend"] = cf_signals

    # ── academic_peer ─────────────────────────────────────────────────────────
    ap_score = 0.0
    ap_signals = []
    if f["academic_hits"] >= 3:
        ap_score += 3.0
        ap_signals.append(f"shares academic content ({f['academic_hits']} academic keywords)")
    if f["academic_url_msgs"] >= 2:
        ap_score += 2.0
        ap_signals.append("shares academic links (course platforms, Google Drive)")
    if f["media_rate"] > HIGH_MEDIA_THRESHOLD:
        ap_score += 1.5
        ap_signals.append(f"high file/media sharing ({f['media_rate']:.0%} of messages)")
    if f["hindi_density"] > HINGLISH_THRESHOLD:
        ap_score += 1.0
        ap_signals.append("Hinglish — likely same cultural background")
    scores["academic_peer"] = ap_score
    signals["academic_peer"] = ap_signals

    # ── family ────────────────────────────────────────────────────────────────
    fam_score = 0.0
    fam_signals = []
    if f["family_hits"] >= 2:
        fam_score += 4.5
        fam_signals.append(f"uses family terms ({f['family_hits']} matches)")
    if f["avg_length"] > 50:
        fam_score += 0.5
        fam_signals.append("longer, more detailed messages (family communication style)")
    scores["family"] = fam_score
    signals["family"] = fam_signals

    # ── colleague ─────────────────────────────────────────────────────────────
    col_score = 0.0
    col_signals = []
    if f["work_hits"] >= 3:
        col_score += 3.0
        col_signals.append(f"work-related vocabulary ({f['work_hits']} keywords)")
    if f["formal_count"] >= 2:
        col_score += 2.5
        col_signals.append(f"uses formal greetings ({f['formal_count']} formal openers)")
    if f["avg_length"] > HIGH_AVG_LENGTH:
        col_score += 1.0
        col_signals.append("long, detailed messages typical of professional context")
    if f["slang_density"] < 0.01:
        col_score += 0.5
        col_signals.append("minimal slang — formal register")
    scores["colleague"] = col_score
    signals["colleague"] = col_signals

    # ── romantic_interest ─────────────────────────────────────────────────────
    rom_score = 0.0
    rom_signals = []
    if f["romantic_hits"] >= 3:
        rom_score += 4.0
        rom_signals.append(f"uses terms of endearment ({f['romantic_hits']} instances)")
    if f["emoji_rate"] > 0.50:
        rom_score += 2.0
        rom_signals.append(f"very high emoji density ({f['emoji_rate']:.0%})")
    if f["late_night_rate"] > 0.30:
        rom_score += 1.5
        rom_signals.append("very frequent late-night messages")
    scores["romantic_interest"] = rom_score
    signals["romantic_interest"] = rom_signals

    # ── acquaintance (baseline) ───────────────────────────────────────────────
    acq_score = 1.0   # always at least 1 — acts as fallback
    acq_signals = ["low engagement or surface-level interaction"]
    if f["total"] < 30:
        acq_score += 1.0
        acq_signals.append(f"low message count ({f['total']} messages)")
    scores["acquaintance"] = acq_score
    signals["acquaintance"] = acq_signals

    return scores, signals