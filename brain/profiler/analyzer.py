"""
brain/profiler/analyzer.py

Takes a list of MessageRecord objects for a specific user and extracts
their communication style, emoji habits, and quirks into a CommunicationProfile.

Usage:
    from brain.profiler.analyzer import build_profile
    profile = build_profile(records, user_name="Rishit")
"""
import re
from collections import Counter
from brain.profiler.models import (
    MessageRecord,
    CommunicationProfile,
    RelationshipProfile,
    StyleProfile,
    EmojiProfile,
)

try:
    import emoji as emoji_lib
    EMOJI_LIB_AVAILABLE = True
except ImportError:
    EMOJI_LIB_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

# Words that signal a message opener (first meaningful word)
FILLER_TOKENS = {"the", "a", "an", "is", "it", "i", "you", "we", "ok",
                 "okay", "and", "but", "so", "just", "also", "then"}

# Common slang / informal tokens — extend this over time
SLANG_VOCAB = {
    "lol", "lmao", "lmfao", "omg", "wtf", "idk", "tbh", "ngl", "imo",
    "imho", "btw", "fyi", "smh", "fr", "frfr", "lowkey", "highkey",
    "bruh", "bro", "dude", "yo", "yoo", "yooo", "nah", "yeah", "yep",
    "yup", "nope", "oof", "rip", "sus", "slay", "vibe", "lit", "fire",
    "noice", "bet", "fax", "cap", "nocap", "sheesh", "brb", "ttyl",
    "lmk", "hmu", "nvm", "gg", "irl", "afk", "smth", "kinda", "sorta",
    "gonna", "wanna", "gotta", "tryna", "cya", "tysm", "ty", "np",
    "bc", "cuz", "tho", "tbt", "haha", "hahaha", "hehe", "lol",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def build_profile(
    records: list[MessageRecord],
    user_name: str,
    platform: str = "",
) -> CommunicationProfile:
    """
    Build a full CommunicationProfile for `user_name` from parsed message records.

    Args:
        records:   All messages from a chat export (multiple senders OK)
        user_name: The sender name to profile (as it appears in the export)
        platform:  "whatsapp" | "telegram" | etc. (inferred from records if empty)

    Returns:
        CommunicationProfile with global style + per-contact breakdown
    """
    if not records:
        raise ValueError("No records provided.")

    # Infer platform if not supplied
    if not platform:
        platform = records[0].platform

    # Split: user's messages vs everyone else
    user_msgs = [r for r in records if r.sender == user_name]
    if not user_msgs:
        raise ValueError(
            f"No messages found for sender '{user_name}'. "
            f"Senders in this export: {sorted({r.sender for r in records})}"
        )

    # Global style across ALL of the user's messages
    global_style = _extract_style(user_msgs)
    global_emoji = _extract_emoji_profile(user_msgs)

    # Per-contact profiles
    contacts = _group_by_contact(records, user_name)
    relationships = []
    for contact_name, contact_records in contacts.items():
        your_msgs_in_chat = [r for r in contact_records if r.sender == user_name]
        if len(your_msgs_in_chat) < 5:
            continue  # not enough data to profile
        rel = RelationshipProfile(
            contact_name=contact_name,
            message_count=len(your_msgs_in_chat),
            style=_extract_style(your_msgs_in_chat),
            emoji_usage=_extract_emoji_profile(your_msgs_in_chat),
            tone_notes=_extract_tone_notes(your_msgs_in_chat),
        )
        relationships.append(rel)

    # Sort relationships by message count descending
    relationships.sort(key=lambda r: r.message_count, reverse=True)

    return CommunicationProfile(
        platform=platform,
        total_messages_analyzed=len(user_msgs),
        global_style=global_style,
        global_emoji=global_emoji,
        relationships=relationships,
    )


# ── Style extraction ───────────────────────────────────────────────────────────

def _extract_style(messages: list[MessageRecord]) -> StyleProfile:
    """Extract writing style patterns from a list of messages."""
    if not messages:
        return StyleProfile()

    # Filter to text-only messages (no media, non-empty)
    text_msgs = [m for m in messages if m.content.strip()]
    if not text_msgs:
        return StyleProfile()

    char_counts = [m.char_count for m in text_msgs]
    word_counts = [m.word_count for m in text_msgs]

    avg_char = sum(char_counts) / len(char_counts)
    avg_word = sum(word_counts) / len(word_counts)

    # Capitalisation: what % of messages start with lowercase?
    lowercase_starts = sum(
        1 for m in text_msgs
        if m.content and m.content[0].islower()
    )
    uses_lowercase_heavily = (lowercase_starts / len(text_msgs)) > 0.6

    # Punctuation: do messages frequently end with . ! ? etc.?
    punct_endings = sum(
        1 for m in text_msgs
        if m.content and m.content[-1] in ".!?,"
    )
    uses_punctuation = (punct_endings / len(text_msgs)) > 0.3

    # Openers: first meaningful word of each message
    openers = _extract_openers(text_msgs)

    # Closers: last meaningful word/token of each message
    closers = _extract_closers(text_msgs)

    # Slang tokens found
    slang = _extract_slang(text_msgs)

    # Quirks (behavioural patterns)
    quirks = _detect_quirks(text_msgs, uses_lowercase_heavily)

    return StyleProfile(
        avg_message_length=round(avg_char, 1),
        avg_word_count=round(avg_word, 1),
        uses_lowercase_heavily=uses_lowercase_heavily,
        uses_punctuation=uses_punctuation,
        common_openers=openers[:8],
        common_closers=closers[:8],
        slang_tokens=slang[:15],
        typing_quirks=quirks,
    )


def _extract_openers(messages: list[MessageRecord]) -> list[str]:
    """Most common first words/phrases."""
    opener_counter: Counter = Counter()
    for m in messages:
        words = m.content.lower().split()
        if not words:
            continue
        first = words[0].strip(".,!?\"'")
        if first and first not in FILLER_TOKENS and len(first) > 1:
            opener_counter[first] += 1
        # Also capture two-word openers
        if len(words) >= 2:
            bigram = f"{first} {words[1].strip('.,!?\"\'')}"
            opener_counter[bigram] += 1
    return [w for w, _ in opener_counter.most_common(20) if _ >= 2]


def _extract_closers(messages: list[MessageRecord]) -> list[str]:
    """Most common last tokens (often reaction words like 'lol', 'haha')."""
    closer_counter: Counter = Counter()
    for m in messages:
        words = m.content.lower().split()
        if not words:
            continue
        last = words[-1].strip(".,!?\"'")
        if last and len(last) > 1:
            closer_counter[last] += 1
    return [w for w, _ in closer_counter.most_common(20) if _ >= 2]


def _extract_slang(messages: list[MessageRecord]) -> list[str]:
    """Slang/informal tokens the user actually uses."""
    found: Counter = Counter()
    for m in messages:
        tokens = set(re.findall(r"[a-z']+", m.content.lower()))
        for token in tokens:
            if token in SLANG_VOCAB:
                found[token] += 1
    return [w for w, _ in found.most_common() if _ >= 2]


def _detect_quirks(messages: list[MessageRecord], uses_lowercase: bool) -> list[str]:
    """Detect consistent typing patterns."""
    quirks: list[str] = []

    if uses_lowercase:
        quirks.append("no caps")

    # Trailing dots (...)
    trailing_dots = sum(1 for m in messages if m.content.endswith("..."))
    if trailing_dots / len(messages) > 0.08:
        quirks.append("trailing dots...")

    # Excessive exclamation marks
    exclamation = sum(1 for m in messages if "!!" in m.content)
    if exclamation / len(messages) > 0.1:
        quirks.append("double !!")

    # Question mark clusters
    q_clusters = sum(1 for m in messages if "??" in m.content)
    if q_clusters / len(messages) > 0.08:
        quirks.append("double ??")

    # Very short messages (≤ 3 words) — texting in bursts
    short = sum(1 for m in messages if m.word_count <= 3)
    if short / len(messages) > 0.5:
        quirks.append("burst texter (many short messages)")

    # Very long messages (≥ 50 words)
    long = sum(1 for m in messages if m.word_count >= 50)
    if long / len(messages) > 0.1:
        quirks.append("sends long messages")

    # Repeated characters (yesssss / noooo)
    repeated_chars = sum(
        1 for m in messages
        if re.search(r'(.)\1{2,}', m.content.lower())
    )
    if repeated_chars / len(messages) > 0.08:
        quirks.append("stretches words (yesss / noooo)")

    return quirks


# ── Emoji extraction ───────────────────────────────────────────────────────────

def _extract_emoji_profile(messages: list[MessageRecord]) -> EmojiProfile:
    """Extract emoji usage patterns."""
    msgs_with_emoji = [m for m in messages if m.has_emoji]
    if not msgs_with_emoji:
        return EmojiProfile()

    all_emojis: list[str] = []

    for m in msgs_with_emoji:
        if EMOJI_LIB_AVAILABLE:
            # Use the emoji library for accurate extraction
            emojis_in_msg = [e["emoji"] for e in emoji_lib.emoji_list(m.content)]
        else:
            # Fallback: basic unicode range scan
            emojis_in_msg = _extract_emojis_fallback(m.content)
        all_emojis.extend(emojis_in_msg)

    counter = Counter(all_emojis)
    total = len(all_emojis)
    unique = len(counter)
    top = counter.most_common(10)
    frequency = total / len(messages) if messages else 0.0

    return EmojiProfile(
        total_emoji_count=total,
        unique_emoji_count=unique,
        top_emojis=top,
        emoji_frequency=round(frequency, 3),
    )


def _extract_emojis_fallback(text: str) -> list[str]:
    """Fallback emoji extractor using unicode ranges."""
    emoji_ranges = [
        (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF),
        (0x1F700, 0x1F77F), (0x2600, 0x26FF), (0x2700, 0x27BF),
        (0x1F900, 0x1F9FF), (0x1FA00, 0x1FA6F), (0x1FA70, 0x1FAFF),
    ]
    result = []
    for char in text:
        cp = ord(char)
        if any(s <= cp <= e for s, e in emoji_ranges):
            result.append(char)
    return result


# ── Tone notes (LLM-free heuristics) ──────────────────────────────────────────

def _extract_tone_notes(messages: list[MessageRecord]) -> list[str]:
    """
    Lightweight heuristic tone observations for a relationship.
    These are human-readable labels — no LLM needed.
    """
    notes: list[str] = []
    total = len(messages)
    if total == 0:
        return notes

    emoji_rate = sum(1 for m in messages if m.has_emoji) / total
    question_rate = sum(1 for m in messages if "?" in m.content) / total
    avg_len = sum(m.char_count for m in messages) / total

    if emoji_rate > 0.4:
        notes.append("emoji-heavy communication")
    elif emoji_rate < 0.05:
        notes.append("rarely uses emoji")

    if question_rate > 0.3:
        notes.append("frequently asks questions")

    if avg_len > 200:
        notes.append("sends detailed/long messages")
    elif avg_len < 30:
        notes.append("very brief, quick-fire style")

    return notes


# ── Grouping helpers ───────────────────────────────────────────────────────────

def _group_by_contact(
    records: list[MessageRecord],
    user_name: str,
) -> dict[str, list[MessageRecord]]:
    """
    Group records by the contact the user is talking to.
    For 1-on-1 chats: the other sender IS the contact.
    For group chats: group by chat_name.
    """
    # Determine unique senders per chat
    chats: dict[str, list[MessageRecord]] = {}
    for r in records:
        chats.setdefault(r.chat_name, []).append(r)

    result: dict[str, list[MessageRecord]] = {}
    for chat_name, msgs in chats.items():
        senders = {m.sender for m in msgs}
        if len(senders) == 2:
            # 1-on-1: contact is whoever isn't the user
            contact = next(s for s in senders if s != user_name)
            result[contact] = msgs
        else:
            # Group chat: use chat name as key
            result[chat_name] = msgs

    return result


# ── Profile persistence ───────────────────────────────────────────────────────

def save_profile(profile: CommunicationProfile, path: str | Path) -> Path:
    """
    Save a CommunicationProfile to a JSON file.

    Why JSON?
    - Human readable — you can open it and verify it makes sense
    - Pydantic serializes it cleanly (model_dump_json handles datetime, tuples, etc.)
    - No database needed for Phase 1
    - Easy to diff in git if you ever want to track profile evolution

    Args:
        profile: The profile to save
        path:    Where to save it. Convention: data/profiles/{name}.json

    Returns:
        The resolved Path where the file was written
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return out


def load_profile(path: str | Path) -> CommunicationProfile:
    """
    Load a CommunicationProfile from a JSON file.

    Args:
        path: Path to the JSON file written by save_profile()

    Returns:
        CommunicationProfile with all fields restored

    Raises:
        FileNotFoundError: if the path doesn't exist
        ValueError: if the file is not valid profile JSON
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Profile not found: {p}\n"
            f"Run setup first: python cli/main.py setup <export_file> --you \"Your Name\""
        )
    try:
        return CommunicationProfile.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Could not parse profile at {p}: {e}") from e


def profile_path(user_name: str, profiles_dir: str | Path = "data/profiles") -> Path:
    """
    Canonical path for a user's profile file.

    Normalises the name (lowercase, spaces → underscores) so
    'Rishit Sharma' and 'rishit sharma' resolve to the same file.
    """
    safe_name = user_name.lower().strip().replace(" ", "_")
    return Path(profiles_dir) / f"{safe_name}.json"