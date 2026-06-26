"""
brain/media/emoji_suggester.py

Context-aware reaction emoji suggestions — fully local, zero API calls.

Algorithm:
  1. Detect message tone from content keywords + relationship type
  2. Build a candidate pool from the tone-appropriate emoji set
  3. Boost emojis that appear in the user's personal emoji profile
  4. Further boost emojis found in the user's reaction history (Telegram)
  5. Return top_k ranked ReactionSuggestion objects

Why not an LLM here?
  This is low-latency UX. The user wants a suggestion in < 200ms
  as they're reading an incoming message. LLM inference takes 3–10s.
  Heuristic matching against the user's actual emoji profile is
  fast, explainable, and surprisingly accurate because we're not
  guessing — we know exactly which emojis this person actually uses.
"""
import re
from collections import Counter
from typing import Optional

from brain.media.models import ReactionSuggestion
from brain.profiler.models import CommunicationProfile, RelationshipProfile


# ══════════════════════════════════════════════════════════════════════════════
# TONE-TO-EMOJI POOLS
# Curated per emotional register. These are defaults before personalisation.
# Position in list = base rank (index 0 is most appropriate).
# ══════════════════════════════════════════════════════════════════════════════

TONE_EMOJIS: dict[str, list[str]] = {
    "humor":       ["😂", "💀", "🤣", "😭", "😅", "🙃", "💅", "🫠"],
    "celebration": ["🎉", "🥳", "🔥", "💯", "🙌", "✨", "🚀", "👏"],
    "affection":   ["❤️", "🥰", "😘", "💕", "🫶", "💗", "🤗", "💙"],
    "emotional":   ["🫂", "❤️", "🙏", "💙", "😢", "🥺", "❤️‍🩹", "💜"],
    "question":    ["🤔", "👀", "😭", "🫠", "💀", "😅", "🤷"],
    "support":     ["💪", "👍", "🙏", "❤️", "🫂", "✅", "🤝"],
    "agreement":   ["👍", "✅", "💯", "👌", "🤙", "🙌", "👊"],
    "neutral":     ["👍", "😊", "🙂", "💯", "🤙", "👌", "🫡"],
}

# ══════════════════════════════════════════════════════════════════════════════
# TONE DETECTION KEYWORDS
# Matched against the lowercased message text (after tokenisation).
# ══════════════════════════════════════════════════════════════════════════════

HUMOR_KEYWORDS = {
    "lol", "lmao", "lmfao", "haha", "hehe", "funny", "joke", "hilarious",
    "dead", "bruh", "😂", "💀", "🤣", "dying", "omg", "noooo", "yikes",
    "rofl", "rip",
}
CELEBRATION_KEYWORDS = {
    "done", "finished", "passed", "got", "selected", "congrats",
    "congratulations", "woohoo", "yay", "awesome", "amazing", "cleared",
    "placed", "offer", "result", "qualify", "achieved", "won", "success",
    "finally", "milestone", "promoted", "scholarship",
}
AFFECTION_KEYWORDS = {
    "love", "miss you", "cute", "sweet", "jaan", "baby", "darling",
    "babe", "muah", "hugs", "miss", "beautiful", "adore", "luv",
}
EMOTIONAL_KEYWORDS = {
    "sad", "cry", "crying", "miss", "tension", "stress", "upset", "hurt",
    "bad day", "kharab", "dukhi", "pareshaan", "worried", "scared",
    "anxious", "depressed", "hard time", "difficult", "struggling",
    "exhausted", "overwhelmed",
}
SUPPORT_KEYWORDS = {
    "help", "stuck", "issue", "problem", "cant", "cannot", "failed",
    "error", "broken", "advice", "suggestion", "struggling", "confused",
    "lost",
}
AGREEMENT_TOKENS = {
    "haan", "yes", "okay", "ok", "sure", "theek", "bilkul", "absolutely",
    "definitely", "agreed", "correct", "right", "exactly", "true",
}

QUESTION_PATTERNS = [
    re.compile(r'\?'),
    re.compile(r'\bkya\b', re.IGNORECASE),
    re.compile(r'\bkyun\b', re.IGNORECASE),
    re.compile(r'\bkab\b', re.IGNORECASE),
    re.compile(r'\bkahan\b', re.IGNORECASE),
    re.compile(r'\bhow\b', re.IGNORECASE),
    re.compile(r'\bwhen\b', re.IGNORECASE),
    re.compile(r'\bwhere\b', re.IGNORECASE),
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def suggest_reactions(
    incoming: str,
    profile: Optional[CommunicationProfile] = None,
    relationship: Optional[RelationshipProfile] = None,
    top_k: int = 6,
) -> list[ReactionSuggestion]:
    """
    Suggest reaction emojis for an incoming message.

    Args:
        incoming:     The message you received
        profile:      User's CommunicationProfile (for personalisation)
        relationship: Relationship context (affects tone interpretation)
        top_k:        Max suggestions to return (default 6)

    Returns:
        Ranked list of ReactionSuggestion objects, most fitting first.
        Always returns a list — empty only if incoming is empty.
    """
    if not incoming or not incoming.strip():
        return []

    tone = _detect_tone(incoming, relationship)
    candidates = _score_candidates(tone, profile)

    # Deduplicate while preserving order (dict insertion order in Python 3.7+)
    seen: set[str] = set()
    unique: list[ReactionSuggestion] = []
    for r in candidates:
        if r.emoji not in seen:
            seen.add(r.emoji)
            unique.append(r)

    return unique[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# TONE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_tone(
    text: str,
    relationship: Optional[RelationshipProfile] = None,
) -> str:
    """
    Detect the dominant emotional register of the incoming message.

    Priority order matters — celebration > affection > humor > emotional
    > support > question > agreement > neutral.

    Relationship type used as a tiebreaker when tone is ambiguous.
    """
    text_l = text.lower()
    words = set(re.findall(r'\w+', text_l))

    # Priority 1: celebration (unambiguous positive events)
    if words & CELEBRATION_KEYWORDS:
        return "celebration"

    # Priority 2: affection (relationship-dependent)
    if words & AFFECTION_KEYWORDS:
        if relationship and relationship.relationship_type in (
            "romantic_interest", "close_friend", "family"
        ):
            return "affection"

    # Priority 3: humor (emoji-heavy check too since 😂 in text)
    if words & HUMOR_KEYWORDS or any(e in text for e in ("😂", "💀", "🤣", "😭")):
        return "humor"

    # Priority 4: emotional distress
    if words & EMOTIONAL_KEYWORDS:
        return "emotional"

    # Priority 5: asking for help/stuck
    if words & SUPPORT_KEYWORDS:
        return "support"

    # Priority 6: question
    if any(p.search(text_l) for p in QUESTION_PATTERNS):
        return "question"

    # Priority 7: agreement / confirmation
    if words & AGREEMENT_TOKENS:
        return "agreement"

    # Default: family often gets warm neutral, others get standard neutral
    return "neutral"


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _score_candidates(
    tone: str,
    profile: Optional[CommunicationProfile],
) -> list[ReactionSuggestion]:
    """
    Build a scored + ranked candidate list.

    Scoring layers (additive):
      Base:           Position in tone pool (0.90 → 0.20 across 8 slots)
      Profile boost:  +0.0–0.50 if emoji appears in user's top_emojis
      Reaction boost: +0.0–0.40 if emoji appears in user's reaction history
    """
    pool = TONE_EMOJIS.get(tone, TONE_EMOJIS["neutral"])

    # Base scores from tone pool position
    base_step = 0.70 / max(len(pool) - 1, 1)
    scores: dict[str, tuple[float, str]] = {
        emoji: (0.90 - i * base_step, "tone_pool")
        for i, emoji in enumerate(pool)
    }

    if profile:
        _apply_profile_boosts(scores, profile)

    # Build result list sorted by final score
    return [
        ReactionSuggestion(
            emoji=emoji,
            confidence=round(min(score, 1.0), 2),
            reason=_build_reason(emoji, tone, source),
            source=source,
        )
        for emoji, (score, source) in sorted(
            scores.items(), key=lambda x: -x[1][0]
        )
    ]


def _apply_profile_boosts(
    scores: dict[str, tuple[float, str]],
    profile: CommunicationProfile,
) -> None:
    """
    Mutate scores dict in-place, applying profile-based boosts.

    Two signals:
    1. top_emojis from messages the user SENT (emoji habits in text)
    2. top_reactions from messages the user REACTED to (Telegram data)
    """
    emoji_data = profile.global_emoji
    if not emoji_data:
        return

    # ── Signal 1: emojis the user sends in text ───────────────────────────────
    if emoji_data.top_emojis:
        max_count = max(c for _, c in emoji_data.top_emojis)
        for emoji, count in emoji_data.top_emojis:
            boost = 0.50 * (count / max_count)
            current_score, current_source = scores.get(emoji, (0.05, "tone_pool"))
            new_score = current_score + boost
            source = "profile" if new_score > current_score else current_source
            scores[emoji] = (new_score, source)

    # ── Signal 2: emojis the user uses as reactions (Telegram only) ───────────
    if emoji_data.top_reactions:
        max_react = max(c for _, c in emoji_data.top_reactions)
        for emoji, count in emoji_data.top_reactions:
            boost = 0.40 * (count / max_react)
            current_score, _ = scores.get(emoji, (0.05, "tone_pool"))
            scores[emoji] = (current_score + boost, "reaction_history")


def _build_reason(emoji: str, tone: str, source: str) -> str:
    """Human-readable reason string for the CLI display."""
    tone_label = tone.replace("_", " ")
    if source == "reaction_history":
        return f"you've reacted with this before · {tone_label} tone"
    if source == "profile":
        return f"in your top emoji · {tone_label} tone"
    return f"fits {tone_label} tone"