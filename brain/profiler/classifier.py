"""
brain/profiler/classifier.py

Relationship classifier — scores ConversationSignals into relationship types.

v4 changes (context-aware rewrite):
  This module is now a THIN SCORING LAYER only.
  All signal extraction has moved to signals.py.

  The previous version computed 12 numeric features from raw word counts
  and compared them to fixed thresholds. This produced accurate results
  only when the vocabulary matched exactly — it could not handle compound
  signals, phrases, behavioral patterns, or topic context.

  This version receives a ConversationSignals object (already extracted
  and structured by signals.py) and scores it into relationship categories.
  Each category has a multi-dimensional evidence accumulation model:
    - vocabulary evidence (still useful, now normalized)
    - phrase evidence (higher weight than vocabulary)
    - behavioral evidence (highest weight — patterns reveal relationship)
    - topic distribution evidence
    - session structure evidence
    - emoji semantic evidence
    - compound signal resolution (overrides ambiguous individual signals)
"""
from brain.profiler.models import MessageRecord
from brain.profiler.signals import ConversationSignals, extract_conversation_signals


# ── Constants ──────────────────────────────────────────────────────────────────

MIN_MESSAGES     = 10
CONFIDENCE_FLOOR = 0.40


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_relationship(
    user_messages: list[MessageRecord],
    all_messages: list[MessageRecord],
) -> tuple[str, float, list[str]]:
    """
    Classify the relationship type from a conversation.

    Returns (relationship_type, confidence, human_readable_signals).
    """
    if len(user_messages) < MIN_MESSAGES:
        return "unknown", 0.0, ["insufficient message count for classification"]

    signals = extract_conversation_signals(user_messages, all_messages)
    scores, evidence = _score_all(signals)

    if not scores:
        return "unknown", 0.0, ["no strong signals detected"]

    best_type  = max(scores, key=scores.get)
    best_score = scores[best_type]
    confidence = round(min(best_score / 10.0, 1.0), 2)

    if confidence < CONFIDENCE_FLOOR:
        return "acquaintance", confidence, evidence.get(best_type, [])

    return best_type, confidence, evidence.get(best_type, [])


def get_all_scores(
    user_messages: list[MessageRecord],
    all_messages: list[MessageRecord],
) -> dict[str, tuple[float, list[str]]]:
    """
    Return scores and evidence for ALL relationship categories.
    Used by the CLI `signals` command.
    """
    if len(user_messages) < MIN_MESSAGES:
        return {}

    signals = extract_conversation_signals(user_messages, all_messages)
    scores, evidence = _score_all(signals)

    return {
        cat: (round(min(score / 10.0, 1.0), 2), evidence.get(cat, []))
        for cat, score in sorted(scores.items(), key=lambda x: -x[1])
        if score > 0
    }


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_all(
    s: ConversationSignals,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    Score all relationship categories from ConversationSignals.

    Scoring hierarchy (highest to lowest weight):
      1. Behavioral patterns (has_planning_behavior, has_checkin_behavior)
      2. Compound signals   (college_friend_compound_count)
      3. Topic distribution (dominant_topic, topic_*)
      4. Session dynamics   (avg_session_depth, sessions_per_week)
      5. Phrase counts      (planning_phrase_count, etc.)
      6. Vocabulary         (slang_density, hindi_density)
      7. Emoji semantics    (humor_emoji_rate, affection_emoji_rate)
    """
    scores:   dict[str, float]      = {}
    evidence: dict[str, list[str]]  = {}

    scores["close_friend"],      evidence["close_friend"]      = _score_close_friend(s)
    scores["college_friend"],    evidence["college_friend"]    = _score_college_friend(s)
    scores["family"],            evidence["family"]            = _score_family(s)
    scores["romantic_interest"], evidence["romantic_interest"] = _score_romantic(s)
    scores["colleague"],         evidence["colleague"]         = _score_colleague(s)
    scores["acquaintance"],      evidence["acquaintance"]      = _score_acquaintance(s)

    return scores, evidence


def _score_close_friend(s: ConversationSignals) -> tuple[float, list[str]]:
    score = 0.0
    ev: list[str] = []

    # ── Behavioral patterns (highest weight) ──────────────────────────────────
    if s.has_planning_behavior:
        score += 3.0
        ev.append(f"plans meetups regularly ({s.planning_phrase_count} planning phrases)")

    if s.has_emotional_support:
        score += 2.5
        ev.append(f"shares emotional concerns ({s.emotional_phrase_count} support exchanges)")

    if s.has_checkin_behavior:
        score += 1.5
        ev.append(f"daily check-ins ({s.checkin_phrase_count} check-in messages)")

    # ── Session dynamics ───────────────────────────────────────────────────────
    if s.avg_session_depth > 20:
        score += 2.5
        ev.append(f"very deep conversations: avg {s.avg_session_depth:.0f} messages/session")
    elif s.avg_session_depth > 10:
        score += 1.5
        ev.append(f"engaged conversations: avg {s.avg_session_depth:.0f} messages/session")

    if s.deep_session_rate > 0.40:
        score += 1.5
        ev.append(f"frequent long sessions: {s.deep_session_rate:.0%} have 10+ messages")

    if s.sessions_per_week > 5:
        score += 2.0
        ev.append(f"very frequent contact: {s.sessions_per_week:.1f} conversations/week")
    elif s.sessions_per_week > 2:
        score += 1.0
        ev.append(f"regular contact: {s.sessions_per_week:.1f} conversations/week")

    # ── Vocabulary signals ─────────────────────────────────────────────────────
    if s.slang_density > 0.04:
        score += 1.5
        if s.slang_top_words:
            ev.append(f"slang: {', '.join(s.slang_top_words[:4])}")

    if s.hindi_density > 0.05:
        score += 1.5
        if s.hindi_top_words:
            ev.append(f"Hinglish: {', '.join(s.hindi_top_words[:4])}")

    # ── Nickname usage (strong friend signal) ─────────────────────────────────
    if s.nickname_hits >= 5:
        score += 1.5
        ev.append(f"uses nicknames/affectionate terms ({s.nickname_hits}×)")

    # ── Emoji: humor dominant ─────────────────────────────────────────────────
    if s.humor_emoji_rate > 0.15:
        score += 1.0
        ev.append(f"humor emojis dominant (😂💀🤣) — {s.humor_emoji_rate:.0%} of messages")

    # ── Topic: social dominant ─────────────────────────────────────────────────
    if s.dominant_topic == "social":
        score += 1.0
        ev.append("conversation is primarily social/casual in nature")

    if s.late_night_rate > 0.20:
        score += 1.0
        ev.append(f"late-night messages: {s.late_night_rate:.0%} after 11pm or before 4am")

    if s.formal_count == 0 and s.session_count > 5:
        score += 0.5
        ev.append("no formal greetings across entire conversation")

    if 0.35 <= s.user_balance <= 0.65:
        score += 0.5
        ev.append(f"balanced conversation ({s.user_balance:.0%} your messages)")

    return score, ev


def _score_college_friend(s: ConversationSignals) -> tuple[float, list[str]]:
    """
    college_friend: close friend who is also a classmate/batchmate.
    Strong compound signal — bhai + assignment in same message.
    Both academic AND social topics present. Hinglish throughout.
    """
    score = 0.0
    ev: list[str] = []

    # ── Compound signals (highest weight for this category) ───────────────────
    if s.college_friend_compound_count >= 3:
        score += 4.0
        ev.append(
            f"college friend compound signal: casual address + academic task "
            f"in same message ({s.college_friend_compound_count}× — e.g. 'bhai assignment bhej de')"
        )
    elif s.college_friend_compound_count >= 1:
        score += 2.0
        ev.append(
            f"college friend compound: casual address + academic task "
            f"({s.college_friend_compound_count}×)"
        )

    # ── Academic task phrases ──────────────────────────────────────────────────
    if s.academic_task_phrase_count >= 3:
        score += 2.5
        ev.append(f"academic collaboration ({s.academic_task_phrase_count} task exchanges: notes, assignments, proxy)")

    # ── College vocabulary ─────────────────────────────────────────────────────
    if s.college_academic_hits >= 4:
        score += 2.0
        ev.append(f"college academic vocabulary: {s.college_academic_hits} distinct terms")

    if s.college_social_hits >= 2:
        score += 1.5
        ev.append(f"college social vocabulary (canteen, hostel, fest, placement)")

    # ── Career terms signal same-batch context ────────────────────────────────
    if s.topic_career > 0.10:
        score += 1.0
        ev.append("discusses career/placement — likely same batch/cohort")

    # ── Planning + Hinglish = college friend ──────────────────────────────────
    if s.has_planning_behavior and s.hindi_density > 0.05:
        score += 1.5
        ev.append("plans meetups in Hinglish — classic college friend pattern")

    # ── Both academic AND social topics present ────────────────────────────────
    if s.topic_academic > 0.15 and s.topic_social > 0.10:
        score += 1.5
        ev.append(
            f"mixed academic+social conversation — "
            f"academic {s.topic_academic:.0%}, social {s.topic_social:.0%}"
        )

    return score, ev


def _score_family(s: ConversationSignals) -> tuple[float, list[str]]:
    """
    family: parent, sibling, relative.
    REQUIRES strong unambiguous family terms (maa, papa, didi — not bhai alone).
    Check-in behavior with family terms is a strong compound signal.
    """
    score = 0.0
    ev: list[str] = []

    # ── Unambiguous family terms (highest weight) ─────────────────────────────
    if s.family_strong_hits >= 3:
        score += 5.0
        ev.append(f"clear family terms used: {s.family_strong_hits} distinct terms (maa, papa, didi, etc.)")
    elif s.family_strong_hits == 2:
        score += 3.5
        ev.append(f"family terms detected ({s.family_strong_hits} distinct terms)")
    elif s.family_strong_hits == 1:
        score += 1.5
        ev.append("one family term detected (weak signal — could be casual address)")

    # ── Family compound signal ─────────────────────────────────────────────────
    if s.family_compound_count >= 2:
        score += 2.0
        ev.append(f"family compound: unambiguous family term + logistics ({s.family_compound_count}×)")

    # ── Check-in behavior = strong family signal ───────────────────────────────
    if s.has_checkin_behavior and s.family_strong_hits >= 1:
        score += 2.5
        ev.append("daily check-ins combined with family terms — parent/sibling pattern")
    elif s.has_checkin_behavior:
        score += 0.5
        ev.append("check-in behavior detected (insufficient family terms to confirm family)")

    # ── Topic logistics dominant ───────────────────────────────────────────────
    if s.topic_logistics > 0.20:
        score += 1.5
        ev.append(f"logistics-heavy conversation: {s.topic_logistics:.0%} about daily whereabouts")

    # ── Family conversations are often imbalanced ─────────────────────────────
    if s.user_balance < 0.35:
        score += 0.5
        ev.append(f"they initiate more ({1 - s.user_balance:.0%} their messages — parent pattern)")

    return score, ev


def _score_romantic(s: ConversationSignals) -> tuple[float, list[str]]:
    score = 0.0
    ev: list[str] = []

    # ── Romantic vocabulary ────────────────────────────────────────────────────
    if s.romantic_hits >= 5:
        score += 4.0
        ev.append(f"terms of endearment used frequently ({s.romantic_hits}×)")
    elif s.romantic_hits >= 2:
        score += 2.0
        ev.append(f"terms of endearment detected ({s.romantic_hits}×)")

    # ── Affection emojis dominant ─────────────────────────────────────────────
    if s.affection_emoji_rate > 0.20:
        score += 2.5
        ev.append(f"affection emojis dominant (❤️🥰😘): {s.affection_emoji_rate:.0%} of messages")

    # ── Emotional intimacy ────────────────────────────────────────────────────
    if s.has_emotional_support and s.topic_emotional > 0.15:
        score += 2.0
        ev.append("deep emotional sharing — high trust, personal disclosures")

    # ── Check-in behavior (daily routine monitoring = high intimacy) ──────────
    if s.has_checkin_behavior:
        score += 1.5
        ev.append("monitors daily routine — strong intimacy indicator")

    # ── Session depth and frequency ────────────────────────────────────────────
    if s.avg_session_depth > 25:
        score += 2.0
        ev.append(f"very long conversations: avg {s.avg_session_depth:.0f} messages/session")

    if s.sessions_per_week > 7:
        score += 1.5
        ev.append(f"daily contact: {s.sessions_per_week:.1f} conversations/week")

    if s.late_night_rate > 0.30:
        score += 1.0
        ev.append(f"frequent late-night messages: {s.late_night_rate:.0%}")

    return score, ev


def _score_colleague(s: ConversationSignals) -> tuple[float, list[str]]:
    score = 0.0
    ev: list[str] = []

    # ── Work vocabulary ────────────────────────────────────────────────────────
    if s.work_hits >= 4:
        score += 3.0
        ev.append(f"work vocabulary: {s.work_hits} distinct work terms")
    elif s.work_hits >= 2:
        score += 1.5
        ev.append(f"some work vocabulary detected ({s.work_hits} terms)")

    # ── Formal register ────────────────────────────────────────────────────────
    if s.formal_count >= 3:
        score += 2.5
        ev.append(f"formal greetings: {s.formal_count} messages start formally")

    # ── Topic: professional dominant ──────────────────────────────────────────
    if s.dominant_topic == "professional":
        score += 2.0
        ev.append(f"professional topics dominate ({s.topic_professional:.0%} of conversation)")

    # ── Long structured messages ───────────────────────────────────────────────
    if s.avg_message_length > 120:
        score += 1.0
        ev.append(f"long structured messages: avg {s.avg_message_length:.0f} chars")

    # ── Low slang ─────────────────────────────────────────────────────────────
    if s.slang_density < 0.01 and s.hindi_density < 0.02:
        score += 0.5
        ev.append("minimal slang and Hinglish — formal register maintained")

    # ── Short focused sessions ─────────────────────────────────────────────────
    if s.avg_session_depth < 5 and s.sessions_per_week < 3:
        score += 0.5
        ev.append(f"task-focused short sessions: avg {s.avg_session_depth:.1f} messages")

    return score, ev


def _score_acquaintance(s: ConversationSignals) -> tuple[float, list[str]]:
    score = 1.0   # baseline — always at least 1
    ev = ["surface-level or low-engagement interaction"]

    if s.session_count < 5:
        score += 1.0
        ev.append(f"very few conversation sessions: {s.session_count}")

    if s.avg_session_depth < 3:
        score += 0.5
        ev.append(f"very short sessions: avg {s.avg_session_depth:.1f} messages")

    if s.sessions_per_week < 0.5:
        score += 0.5
        ev.append(f"infrequent contact: {s.sessions_per_week:.1f} conversations/week")

    return score, ev