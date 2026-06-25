"""
brain/generator/reply.py

Phase 2B — Reply draft generator.

Fix history:
  v2: Added _strip_thinking() for <think> block stripping.
  v3: chat() → generate(). Wrong theory, same 0-char issue.
  v4: Removed num_predict cap + think=False. WORKS. Output was mechanical.
  v5: Overcomplicated fallback chain broke v4's working generate() call.
      REVERTED to v4 LLM call exactly. Prompt improved only.

v5 final = v4 LLM call + v5 prompt.

Prompt changes from v4:
  - Removed slang word list ("Uses these words: lol, fr, ngl...")
    This caused the model to force-fit vocabulary regardless of context.
  - Replaced with register description: "Casual Hinglish, college student vibe"
  - Added context-analysis step: understand what the message is asking
  - RAG examples now teach style instead of a keyword checklist
  - "Natural > technically correct vocabulary. Don't force words."
"""
import json
import os
import re
from pathlib import Path
from typing import Optional

from brain.profiler.models import CommunicationProfile, RelationshipProfile

DEFAULT_MODEL      = "qwen3.5:9b"
DEFAULT_NUM_DRAFTS = 3
MAX_RAG_EXAMPLES   = 5


class ReplyGenerator:

    def __init__(self, store, profiles_dir: str | Path = "data/profiles", model: str = DEFAULT_MODEL):
        self.store        = store
        self.profiles_dir = Path(profiles_dir)
        self.model        = model
        self._last_raw    = ""

    def generate(self, incoming: str, from_contact: str, user_name: str, num_drafts: int = DEFAULT_NUM_DRAFTS) -> list[str]:
        profile      = self._load_profile_safe(user_name)
        relationship = self._find_relationship(profile, from_contact) if profile else None
        examples     = self._retrieve_examples(incoming, user_name, from_contact)
        system, user = build_prompt(
            incoming=incoming, from_contact=from_contact,
            profile=profile, relationship=relationship,
            examples=examples, num_drafts=num_drafts,
        )
        return self._call_ollama(system, user, num_drafts)

    def _load_profile_safe(self, user_name: str) -> Optional[CommunicationProfile]:
        try:
            from brain.profiler.analyzer import load_profile, profile_path
            return load_profile(profile_path(user_name, self.profiles_dir))
        except (FileNotFoundError, ValueError):
            return None

    def _find_relationship(self, profile: CommunicationProfile, contact_name: str) -> Optional[RelationshipProfile]:
        if not profile or not profile.relationships:
            return None
        cl = contact_name.lower()
        for rel in profile.relationships:
            if rel.contact_name.lower() == cl:
                return rel
        for rel in profile.relationships:
            if cl in rel.contact_name.lower() or rel.contact_name.lower() in cl:
                return rel
        return None

    def _retrieve_examples(self, incoming: str, user_name: str, from_contact: str) -> list[dict]:
        try:
            if not self.store or self.store.count() == 0:
                return []
            return self.store.retrieve_similar(
                text=incoming, top_k=MAX_RAG_EXAMPLES,
                sender_filter=user_name,
                chat_filter=from_contact if from_contact else None,
            )
        except Exception:
            return []

    def _call_ollama(self, system_prompt: str, user_prompt: str, num_drafts: int) -> list[str]:
        """
        Exact same LLM call as v4 — the only approach proven to work.

        think=False  : disables qwen3 reasoning chain entirely. Without this,
                       thinking tokens consume the entire generation budget and
                       response.response is empty.
        no num_predict: no token cap. summary command (proven working) also sets none.
        think= TypeError: older library versions don't support this param — fallback.
        response access: mirrors summary command exactly (.response attribute + .get fallback).
        """
        import ollama

        combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        try:
            try:
                response = ollama.generate(
                    model=self.model, prompt=combined_prompt,
                    stream=False, think=False,
                    options={"temperature": 0.85},
                )
            except TypeError:
                response = ollama.generate(
                    model=self.model, prompt=combined_prompt,
                    stream=False, options={"temperature": 0.85},
                )

            if isinstance(response, dict):
                raw = (response.get("response", "") or "").strip()
            else:
                raw = (getattr(response, "response", None) or "").strip()
                if not raw and hasattr(response, "__getitem__"):
                    raw = (response.get("response", "") or "").strip()

            self._last_raw = raw

            if os.getenv("PERSONA_DEBUG"):
                import sys
                print(f"\n[DEBUG] Response type: {type(response).__name__}", file=sys.stderr)
                print(f"[DEBUG] Raw LLM response ({len(raw)} chars):\n{raw}\n", file=sys.stderr)

            return _parse_reply_json(raw, num_drafts)

        except Exception as e:
            raise RuntimeError(
                f"Ollama call failed ({type(e).__name__}): {e}\n"
                "Is Ollama running?  Run: ollama serve"
            ) from e


def build_prompt(
    incoming: str, from_contact: str,
    profile: Optional[CommunicationProfile],
    relationship: Optional[RelationshipProfile],
    examples: list[dict], num_drafts: int = DEFAULT_NUM_DRAFTS,
) -> tuple[str, str]:
    system = _build_system_prompt(profile, relationship, from_contact, num_drafts)
    user   = _build_user_prompt(incoming, from_contact, examples, num_drafts)
    return system, user


def _build_system_prompt(
    profile: Optional[CommunicationProfile],
    relationship: Optional[RelationshipProfile],
    from_contact: str, num_drafts: int,
) -> str:
    parts: list[str] = []

    rel_label = _rel_label(relationship)
    parts.append(
        f"You help draft WhatsApp replies for someone texting their {rel_label}.\n"
        f"Write what this person would ACTUALLY say — natural and authentic, "
        f"not a caricature of their vocabulary."
    )

    if profile and profile.global_style:
        style = profile.global_style
        emoji = profile.global_emoji

        lines = ["THEIR COMMUNICATION STYLE:"]

        if style.avg_message_length < 50:
            lines.append("- Short punchy messages, sometimes in rapid bursts")
        elif style.avg_message_length < 100:
            lines.append("- Short-to-medium messages, conversational")
        else:
            lines.append("- Medium length, engaged texter")

        if style.uses_lowercase_heavily and not style.uses_punctuation:
            lines.append("- Casual Hinglish — relaxed lowercase, minimal punctuation, sounds like a real college student texting")
        elif style.uses_lowercase_heavily:
            lines.append("- Lowercase, relaxed texting style")

        if emoji and emoji.emoji_frequency > 0.25:
            lines.append(f"- Uses emoji naturally in about {emoji.emoji_frequency:.0%} of messages")
        elif emoji and emoji.emoji_frequency > 0.05:
            lines.append("- Occasional emoji, only when it fits")
        else:
            lines.append("- Rarely uses emoji")

        lines.append(
            "- Study the examples below to match tone. "
            "Do NOT force specific slang — only use words that sound natural for this reply."
        )
        parts.append("\n".join(lines))

    if relationship:
        rel_lines = [f"RELATIONSHIP WITH {from_contact.upper()}:"]
        rel_lines.append(f"- Type: {relationship.relationship_type.replace('_', ' ')}")
        if relationship.tone_notes:
            rel_lines.append(f"- Dynamic: {', '.join(relationship.tone_notes[:2])}")
        parts.append("\n".join(rel_lines))

    parts.append(
        f"TASK: Write {num_drafts} reply options.\n"
        f"Step 1: Understand what {from_contact} is actually asking or saying.\n"
        f"Step 2: Write what this person would naturally reply — consider a yes, a no, and a third angle.\n"
        f"Step 3: Each option max 1-3 short messages. Natural > forced vocabulary.\n"
        f"Output ONLY this JSON:\n"
        f'  {{"replies": ["option1", "option2", "option3"]}}'
    )

    return "\n\n".join(parts)


def _build_user_prompt(incoming: str, from_contact: str, examples: list[dict], num_drafts: int) -> str:
    parts: list[str] = []

    if examples:
        ex_lines = ["PAST MESSAGES (how this person actually writes — match this tone, not specific words):"]
        for ex in examples[:MAX_RAG_EXAMPLES]:
            ex_lines.append(f'  "{ex["content"]}"')
        parts.append("\n".join(ex_lines))

    parts.append(f'{from_contact} sent: "{incoming}"\n\nWrite {num_drafts} reply options as JSON.')
    return "\n\n".join(parts)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_reply_json(raw: str, expected: int = DEFAULT_NUM_DRAFTS) -> list[str]:
    if not raw:
        return []
    text = _strip_thinking(raw)
    if not text:
        return []
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    try:
        data = json.loads(text)
        replies = _extract_replies_from_json(data)
        if replies:
            return replies[:expected]
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"(?:replies|options|drafts)"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if match:
        try:
            replies = _extract_replies_from_json(json.loads(match.group()))
            if replies:
                return replies[:expected]
        except json.JSONDecodeError:
            pass

    numbered = re.findall(r'^\d+\.\s+(.+)$', text, re.MULTILINE)
    if numbered:
        return [n.strip().strip('"') for n in numbered[:expected]]

    quoted = [q for q in re.findall(r'"([^"]{4,200})"', text) if not _is_json_key(q)]
    if quoted:
        return quoted[:expected]

    return []


def _extract_replies_from_json(data) -> list[str]:
    if isinstance(data, dict):
        for key in ("replies", "options", "drafts"):
            if key in data and isinstance(data[key], list):
                return [str(r).strip() for r in data[key] if str(r).strip()]
    if isinstance(data, list):
        return [str(r).strip() for r in data if str(r).strip()]
    return []


def _is_json_key(s: str) -> bool:
    return s in {"replies", "options", "drafts", "reply", "option", "draft"}


def _rel_label(relationship: Optional[RelationshipProfile]) -> str:
    if not relationship:
        return "friend"
    return {
        "close_friend":      "close friend",
        "college_friend":    "college classmate and close friend",
        "family":            "family member",
        "romantic_interest": "partner",
        "colleague":         "colleague",
        "academic_peer":     "academic peer",
        "acquaintance":      "acquaintance",
        "unknown":           "friend",
    }.get(relationship.relationship_type, "friend")