"""
brain/profiler/classifier.py

Infers the relationship type between a user and a contact
from message data alone — no LLM required.

v2 changes:
    _extract_features() now returns full Counter objects (top_slang,
    top_hindi, etc.) so _score_categories() can include the SPECIFIC
    WORDS that triggered each signal, not just category-level densities.

    Example output change:
      Before: "heavy slang usage (4.2% of words)"
      After:  "slang words: bro (23×), yaar (18×), lmao (12×), oye (9×)"
"""
import re
from collections import Counter
from brain.profiler.models import MessageRecord


# ── Tunable thresholds ─────────────────────────────────────────────────────────

MIN_MESSAGES = 10
LATE_NIGHT_START = 23
LATE_NIGHT_END = 4
HIGH_SLANG_THRESHOLD = 0.04
HIGH_EMOJI_THRESHOLD = 0.30
HIGH_MEDIA_THRESHOLD = 0.35
LOW_AVG_LENGTH = 35
HIGH_AVG_LENGTH = 120
HINGLISH_THRESHOLD = 0.06
CONFIDENCE_FLOOR = 0.40
WORD_EVIDENCE_LIMIT = 4     # max words to show per signal


# ── Vocabulary sets ────────────────────────────────────────────────────────────

SLANG_WORDS = {
    "lol", "lmao", "lmfao", "omg", "wtf", "idk", "tbh", "ngl", "imo",
    "bruh", "bro", "dude", "yo", "nah", "yep", "yup", "oof", "rip",
    "bet", "fr", "frfr", "lowkey", "highkey", "slay", "vibe", "lit",
    "noice", "sheesh", "gg", "smh", "hmu", "nvm", "lmk", "bc", "cuz",
    "tho", "gonna", "wanna", "gotta", "tryna", "haha", "hehe",
    "u", "ur", "r", "rn", "ik", "ikr",
}

HINDI_WORDS = {
    "yaar", "bhai", "oye", "arre", "haan", "nahi", "nhi", "kya",
    "toh", "aur", "bhi", "abhi", "thoda", "zyada", "bohot", "bahut",
    "accha", "theek", "sahi", "bilkul", "matlab", "agar", "phir",
    "mein", "mai", "tera", "mera", "uska", "humara", "hnn", "hai",
    "tha", "the", "thi", "rha", "rhe", "rhaa", "kr", "kar", "karo",
    "bhej", "bhejio", "dekh", "sun", "bol", "bata", "aa", "jaa",
    "dekha", "suna", "bola", "hua", "hui", "hue",
}

ACADEMIC_KEYWORDS = {
    "assignment", "homework", "project", "deadline", "submission",
    "submit", "professor", "prof", "lecture", "class", "course",
    "exam", "test", "quiz", "marks", "grade", "semester", "pdf",
    "notes", "study", "lab", "presentation", "ppt", "report",
    "affinity", "empathy", "scamper", "brainstorm", "research",
    "college", "university", "batch", "roll", "enrollment",
    "newtonschool", "playground", "code",
}

WORK_KEYWORDS = {
    "meeting", "client", "office", "email", "agenda", "deliverable",
    "review", "feedback", "sprint", "standup", "budget", "invoice",
    "contract", "proposal", "quarterly", "kpi", "milestone", "manager",
    "team", "colleague", "hr", "onboarding", "interview",
}

FAMILY_TERMS = {
    "mom", "mum", "maa", "mama", "dad", "papa", "paa", "bhai",
    "didi", "bhaiya", "sis", "sister", "brother", "uncle", "aunty",
    "auntie", "nana", "nani", "dada", "dadi", "grandma", "grandpa",
    "family", "ghar", "home", "gharpe", "gharwale",
}

ROMANTIC_TERMS = {
    "love", "babe", "baby", "darling", "sweetheart", "honey",
    "miss you", "missyou", "luv", "cute", "beautiful", "handsome",
}

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
        signals are word-level: e.g. "slang words: bro (23×), yaar (18×)"
    """
    if len(user_messages) < MIN_MESSAGES:
        return "unknown", 0.0, ["insufficient message count for classification"]

    features = _extract_features(user_messages, all_messages)
    scores, signals = _score_categories(features)

    if not scores:
        return "unknown", 0.0, ["no strong signals detected"]

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    confidence = round(min(best_score / 10.0, 1.0), 2)

    if confidence < CONFIDENCE_FLOOR:
        return "acquaintance", confidence, signals.get(best_type, [])

    return best_type, confidence, signals.get(best_type, [])


def get_all_scores(
    user_messages: list[MessageRecord],
    all_messages: list[MessageRecord],
) -> dict[str, tuple[float, list[str]]]:
    """
    Return scores and signals for ALL relationship categories.
    Used by the CLI `signals` command to show full breakdown.

    Returns:
        {category: (confidence, signals_list)}
    """
    if len(user_messages) < MIN_MESSAGES:
        return {}

    features = _extract_features(user_messages, all_messages)
    scores, signals = _score_categories(features)

    return {
        cat: (round(min(score / 10.0, 1.0), 2), signals.get(cat, []))
        for cat, score in sorted(scores.items(), key=lambda x: -x[1])
        if score > 0
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_words(counter: Counter, limit: int = WORD_EVIDENCE_LIMIT) -> str:
    """Format top words from a Counter as 'word (N×), word (N×)'."""
    return ", ".join(
        f"{w} ({c}\u00d7)"
        for w, c in counter.most_common(limit)
        if c >= 2
    )


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_features(
    user_msgs: list[MessageRecord],
    all_msgs: list[MessageRecord],
) -> dict:
    """
    Compute features. Returns both numeric rates AND Counter objects
    so signals can show the specific words that triggered each score.
    """
    total = len(user_msgs)
    text_msgs = [m for m in user_msgs if m.content.strip()]

    # Word-level counters — these are the new addition
    slang_counter:    Counter = Counter()
    hindi_counter:    Counter = Counter()
    academic_counter: Counter = Counter()
    work_counter:     Counter = Counter()
    family_counter:   Counter = Counter()
    romantic_counter: Counter = Counter()

    total_words = 0

    for m in text_msgs:
        words = m.content.lower().split()
        total_words += len(words)
        for word in words:
            w = word.strip(".,!?\"';:()[]")
            if w in SLANG_WORDS:    slang_counter[w] += 1
            if w in HINDI_WORDS:    hindi_counter[w] += 1
            if w in ACADEMIC_KEYWORDS: academic_counter[w] += 1
            if w in WORK_KEYWORDS:  work_counter[w] += 1
            if w in FAMILY_TERMS:   family_counter[w] += 1
            if w in ROMANTIC_TERMS: romantic_counter[w] += 1

    total_words = total_words or 1

    # Rate features
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
    formal_count = sum(
        1 for m in text_msgs
        if FORMAL_GREETINGS.match(m.content)
    )
    academic_url_msgs = sum(
        1 for m in text_msgs
        if any(kw in m.content.lower() for kw in
               ["newtonschool", "classroom", "drive.google", "forms.gle"])
    )

    return {
        "total": total,
        "total_words": total_words,
        "emoji_rate": emoji_rate,
        "media_rate": media_rate,
        "late_night_rate": late_night_rate,
        "avg_length": avg_length,
        "formal_count": formal_count,
        "academic_url_msgs": academic_url_msgs,
        # Density rates (for threshold comparisons)
        "slang_density":    sum(slang_counter.values())    / total_words,
        "hindi_density":    sum(hindi_counter.values())    / total_words,
        # Full counters (for word-level evidence in signals)
        "slang_counter":    slang_counter,
        "hindi_counter":    hindi_counter,
        "academic_counter": academic_counter,
        "work_counter":     work_counter,
        "family_counter":   family_counter,
        "romantic_counter": romantic_counter,
    }


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_categories(f: dict) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    Score each relationship category. Signals now include specific words.
    """
    scores: dict[str, float] = {}
    signals: dict[str, list[str]] = {}

    # ── close_friend ──────────────────────────────────────────────────────────
    cf = 0.0
    cf_s: list[str] = []

    if f["slang_density"] > HIGH_SLANG_THRESHOLD:
        cf += 2.5
        words = _fmt_words(f["slang_counter"])
        cf_s.append(f"slang words: {words}" if words else
                    f"heavy slang ({f['slang_density']:.1%} of words)")

    if f["hindi_density"] > HINGLISH_THRESHOLD:
        cf += 2.0
        words = _fmt_words(f["hindi_counter"])
        cf_s.append(f"Hinglish words: {words}" if words else
                    "Hindi-English code-switching detected")

    if f["late_night_rate"] > 0.20:
        cf += 2.0
        cf_s.append(
            f"late-night messages: {f['late_night_rate']:.0%} of messages "
            f"sent after 11pm or before 4am"
        )

    if f["emoji_rate"] > HIGH_EMOJI_THRESHOLD:
        cf += 1.0
        cf_s.append(f"high emoji usage: {f['emoji_rate']:.0%} of messages have emoji")

    if f["avg_length"] < LOW_AVG_LENGTH:
        cf += 1.0
        cf_s.append(f"short casual messages: avg {f['avg_length']:.0f} chars")

    if f["formal_count"] == 0 and f["total"] > 20:
        cf += 0.5
        cf_s.append("no formal greetings detected across conversation")

    scores["close_friend"] = cf
    signals["close_friend"] = cf_s

    # ── academic_peer ─────────────────────────────────────────────────────────
    ap = 0.0
    ap_s: list[str] = []

    if len(f["academic_counter"]) >= 3:
        ap += 3.0
        words = _fmt_words(f["academic_counter"])
        ap_s.append(f"academic keywords: {words}" if words else
                    f"{len(f['academic_counter'])} academic keywords found")

    if f["academic_url_msgs"] >= 2:
        ap += 2.0
        ap_s.append(
            f"shared academic platform links "
            f"({f['academic_url_msgs']} messages with course/drive links)"
        )

    if f["media_rate"] > HIGH_MEDIA_THRESHOLD:
        ap += 1.5
        ap_s.append(
            f"high file sharing: {f['media_rate']:.0%} of messages are media/files"
        )

    if f["hindi_density"] > HINGLISH_THRESHOLD:
        ap += 1.0
        ap_s.append("Hinglish suggests shared cultural/educational background")

    scores["academic_peer"] = ap
    signals["academic_peer"] = ap_s

    # ── family ────────────────────────────────────────────────────────────────
    fam = 0.0
    fam_s: list[str] = []

    if len(f["family_counter"]) >= 2:
        fam += 4.0
        words = _fmt_words(f["family_counter"])
        fam_s.append(f"family terms used: {words}" if words else
                     f"{len(f['family_counter'])} distinct family terms found")

    if f["avg_length"] > 50:
        fam += 0.5
        fam_s.append(f"longer messages: avg {f['avg_length']:.0f} chars (family style)")

    scores["family"] = fam
    signals["family"] = fam_s

    # ── colleague ─────────────────────────────────────────────────────────────
    col = 0.0
    col_s: list[str] = []

    if len(f["work_counter"]) >= 3:
        col += 3.0
        words = _fmt_words(f["work_counter"])
        col_s.append(f"work vocabulary: {words}" if words else
                     f"{len(f['work_counter'])} work keywords")

    if f["formal_count"] >= 2:
        col += 2.5
        col_s.append(
            f"formal openers detected: {f['formal_count']} messages start formally"
        )

    if f["avg_length"] > HIGH_AVG_LENGTH:
        col += 1.0
        col_s.append(f"long messages: avg {f['avg_length']:.0f} chars (professional pattern)")

    if f["slang_density"] < 0.01:
        col += 0.5
        col_s.append("minimal slang — formal register maintained throughout")

    scores["colleague"] = col
    signals["colleague"] = col_s

    # ── romantic_interest ─────────────────────────────────────────────────────
    rom = 0.0
    rom_s: list[str] = []

    romantic_hits = sum(f["romantic_counter"].values())
    if romantic_hits >= 3:
        rom += 4.0
        words = _fmt_words(f["romantic_counter"])
        rom_s.append(f"terms of endearment: {words}" if words else
                     f"{romantic_hits} romantic terms detected")

    if f["emoji_rate"] > 0.50:
        rom += 2.0
        rom_s.append(f"very high emoji density: {f['emoji_rate']:.0%} of messages")

    if f["late_night_rate"] > 0.30:
        rom += 1.5
        rom_s.append(
            f"very frequent late-night messages: {f['late_night_rate']:.0%}"
        )

    scores["romantic_interest"] = rom
    signals["romantic_interest"] = rom_s

    # ── acquaintance (fallback baseline) ─────────────────────────────────────
    acq = 1.0
    acq_s = ["low or surface-level engagement pattern"]
    if f["total"] < 30:
        acq += 1.0
        acq_s.append(f"low message count: {f['total']} messages total")

    scores["acquaintance"] = acq
    signals["acquaintance"] = acq_s

    return scores, signals