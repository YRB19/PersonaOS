"""brain/generator/reply.py
Phase 2B — Reply draft generator.
Given an incoming message, this module:
  1. Loads the user's CommunicationProfile (style + relationships)
  2. Retrieves the most semantically similar past messages from PersonaStore (RAG)
  3. Builds a structured prompt (system + user, chat-format LLM calls)
  4. Calls Ollama to generate N draft replies that match the user's style
  5. Returns list[str] — the user picks one. Nothing auto-sends.

Architecture:
  ReplyGenerator      ← public class, orchestrates the full pipeline
  build_prompt()      ← public, returns (system, user) tuple — testable without Ollama
  _strip_thinking()   ← strips qwen3 <think>...</think> blocks before parsing
  _parse_reply_json() ← paranoid parser with 4 fallback layers
  _rel_label()        ← maps RelationshipProfile type to human label

Fix history:
  v2: Added _strip_thinking() — qwen3.5:9b outputs <think>...</think> blocks
      before the actual JSON. These blocks contain braces that break the
      [^{}]* regex in layer 2. Stripping them first fixes empty-draft bug.
      Removed /no_think from user message — it is not a standard Ollama
      API parameter and can confuse the model in chat format.
"""

# attached: Ollama ollama qwen qwen3 qwen3.5 Priya,Babbar unparseable
# These tokens are project-specific terms and should not be treated as bugs by
# linters or reviewers. Present to reduce false-positive "unknown word" flags.

import json
import os
import re
from pathlib import Path
from typing import Optional
from brain.profiler.models import CommunicationProfile, RelationshipProfile

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "gemma4:latest"
DEFAULT_NUM_DRAFTS = 3
MAX_RAG_EXAMPLES   = 5

# ── Main class ─────────────────────────────────────────────────────────────────
class ReplyGenerator:
    """
    Generates personalized reply drafts using RAG + style profile.
    Usage:
        from brain.memory.store import PersonaStore
        from brain.generator.reply import ReplyGenerator
        store = PersonaStore("data/db/chroma")
        gen   = ReplyGenerator(store, profiles_dir="data/profiles")
        drafts = gen.generate(
            incoming="are you free tonight?",
            from_contact="Priya",
            user_name="Rishit Babbar",
        )
        for i, draft in enumerate(drafts, 1):
            print(f"{i}. {draft}")
    """
    def __init__(
        self,
        store,
        profiles_dir: str | Path = "data/profiles",
        model: str = DEFAULT_MODEL,
    ):
        self.store        = store
        self.profiles_dir = Path(profiles_dir)
        self.model        = model
        self._last_raw    = ""   # stores last raw LLM response — inspect after a call

    def generate(
        self,
        incoming:     str,
        from_contact: str,
        user_name:    str,
        num_drafts:   int = DEFAULT_NUM_DRAFTS,
    ) -> list[str]:
        """
        Generate personalized reply drafts for an incoming message.
        Args:
            incoming:     The message you received and need to reply to
            from_contact: Who sent it (matched against profile relationship names)
            user_name:    Your display name as it appears in the export and profile
            num_drafts:   Number of options to generate (default 3)
        Returns:
            List of draft strings, typically num_drafts long.
            Empty list only if Ollama returns something completely unparseable.
        Raises:
            RuntimeError: if Ollama is unreachable or returns an API error
        """
        profile      = self._load_profile_safe(user_name)
        relationship = self._find_relationship(profile, from_contact) if profile else None
        examples     = self._retrieve_examples(incoming, user_name, from_contact)

        system, user = build_prompt(
            incoming=incoming,
            from_contact=from_contact,
            profile=profile,
            relationship=relationship,
            examples=examples,
            num_drafts=num_drafts,
        )

        return self._call_ollama(system, user, num_drafts)

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _load_profile_safe(self, user_name: str) -> Optional[CommunicationProfile]:
        try:
            from brain.profiler.analyzer import load_profile, profile_path
            return load_profile(profile_path(user_name, self.profiles_dir))
        except (FileNotFoundError, ValueError):
            return None

    def _find_relationship(
        self,
        profile: CommunicationProfile,
        contact_name: str,
    ) -> Optional[RelationshipProfile]:
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

    def _retrieve_examples(
        self,
        incoming:     str,
        user_name:    str,
        from_contact: str,
    ) -> list[dict]:
        try:
            if not self.store or self.store.count() == 0:
                return []
            return self.store.retrieve_similar(
                text=incoming,
                top_k=MAX_RAG_EXAMPLES,
                sender_filter=user_name,
                chat_filter=from_contact if from_contact else None,
            )
        except Exception:
            return []

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt:   str,
        num_drafts:    int,
    ) -> list[str]:
        """
        Call Ollama using chat format and parse the response.
        Why chat over generate?
          System/user role separation produces more reliable JSON output.
        Debug mode:
          Set PERSONA_DEBUG=1 in environment to print raw LLM output to stderr.
          raw response is also stored on self._last_raw for programmatic access.
        v2 note:
          Removed /no_think from user message. It is not a standard Ollama
          parameter and caused model confusion in chat mode. Thinking blocks
          are now stripped in _parse_reply_json() instead.
        """
        import ollama
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                options={
                    "temperature": 0.85,
                    "num_predict": 512,
                },
            )
            raw = response.message.content.strip()
            self._last_raw = raw

            if os.getenv("PERSONA_DEBUG"):
                import sys
                print(f"\n[DEBUG] Raw LLM response ({len(raw)} chars):\n{raw}\n", file=sys.stderr)

            return _parse_reply_json(raw, num_drafts)
        except Exception as e:
            raise RuntimeError(
                f"Ollama call failed ({type(e).__name__}): {e}\n"
                "Is Ollama running?  Run: ollama serve"
            ) from e

# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(
    incoming:     str,
    from_contact: str,
    profile:      Optional[CommunicationProfile],
    relationship: Optional[RelationshipProfile],
    examples:     list[dict],
    num_drafts:   int = DEFAULT_NUM_DRAFTS,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for reply generation.
    Returns a 2-tuple:
      system → WHO the user is (style + relationship) — stable per contact
      user   → WHAT to reply to (RAG examples + incoming message) — changes each call
    Public so tests can verify prompt content without Ollama.
    """
    system = _build_system_prompt(profile, relationship, from_contact, num_drafts)
    user   = _build_user_prompt(incoming, from_contact, examples, num_drafts)
    return system, user

def _build_system_prompt(
    profile:      Optional[CommunicationProfile],
    relationship: Optional[RelationshipProfile],
    from_contact: str,
    num_drafts:   int,
) -> str:
    parts: list[str] = []
    rel_label = _rel_label(relationship)

    parts.append(
        f"You help draft casual WhatsApp replies for someone texting their {rel_label}.\n"
        f"Write EXACTLY like this person — their slang, their grammar, their emoji habits.\n"
        f"Do NOT be formal. Do NOT sound like an AI. Real person, real texting style."
    )

    if profile and profile.global_style:
        style = profile.global_style
        emoji = profile.global_emoji
        lines = ["THEIR WRITING STYLE (match this exactly):"]
        
        if style.avg_message_length < 50:
            lines.append("- Very short messages: 5 to 20 words maximum")
        elif style.avg_message_length < 100:
            lines.append(f"- Short messages: roughly {style.avg_message_length:.0f} chars")
        else:
            lines.append(f"- Medium length: roughly {style.avg_message_length:.0f} chars")

        if style.uses_lowercase_heavily:
            lines.append("- Never capitalizes — always lowercase")
        if not style.uses_punctuation:
            lines.append("- Minimal punctuation — no periods, rare commas")
        if style.slang_tokens:
            lines.append(f"- Uses these words: {', '.join(style.slang_tokens[:7])}")
        if style.typing_quirks:
            lines.append(f"- Quirks: {', '.join(style.typing_quirks[:3])}")
        if emoji and emoji.emoji_frequency > 0.1:
            top = " ".join(e for e, _ in emoji.top_emojis[:4])
            lines.append(f"- About {emoji.emoji_frequency:.1f} emoji per message, favorites: {top}")
        else:
            lines.append("- Rarely or never uses emoji")
        parts.append("\n".join(lines))

    if relationship:
        rel_lines = [f"CONTEXT WITH {from_contact.upper()}:"]
        rel_lines.append(f"- Relationship: {relationship.relationship_type.replace('_', ' ')}")
        if relationship.tone_notes:
            rel_lines.append(f"- Tone: {', '.join(relationship.tone_notes[:2])}")
        parts.append("\n".join(rel_lines))

    parts.append(
        f"OUTPUT: exactly {num_drafts} reply options. Rules:\n"
        f"- 1 to 3 short messages per option (they often text in bursts)\n"
        f"- Vary the options: one brief, one more engaged, one with personality\n"
        f"- Match their vocabulary, capitalization, and emoji habits exactly\n"
        f"- Output ONLY this JSON, nothing else:\n"
        f'  {{"replies": ["option1", "option2", "option3"]}}'
    )

    return "\n\n".join(parts)

def _build_user_prompt(
    incoming:     str,
    from_contact: str,
    examples:     list[dict],
    num_drafts:   int,
) -> str:
    parts: list[str] = []
    
    if examples:
        ex_lines = ["PAST MESSAGES IN SIMILAR CONTEXTS (imitate this style):"]
        for ex in examples[:MAX_RAG_EXAMPLES]:
            ex_lines.append(f'  "{ex["content"]}"')
        parts.append("\n".join(ex_lines))

    parts.append(
        f'{from_contact} sent: "{incoming}"\n\n'
        f'Write {num_drafts} reply options as JSON.'
    )

    return "\n\n".join(parts)

# ── Output parsing ─────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """
    Strip qwen3-style <think>...</think> blocks from LLM output.

    Why this exists:
      qwen3 and qwen3.5 models output their internal reasoning between
      <think> and </think> tags before the actual response. Without stripping
      these blocks, the JSON regex patterns fail because thinking blocks
      routinely contain { } characters that break the [^{}]* pattern in
      layer-2 parsing. The actual JSON is always AFTER the closing </think>.

    Example raw output from qwen3.5:9b:
      <think>
      The user wants me to draft replies. Let me think about how Rishit
      would respond... considering his style with {slang} and {tone}...
      </think>

      {"replies": ["haan yaar", "lol ok", "aaja bhai"]}

    This function reduces that to just:
      {"replies": ["haan yaar", "lol ok", "aaja bhai"]}
    """
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return stripped.strip()


def _parse_reply_json(raw: str, expected: int = DEFAULT_NUM_DRAFTS) -> list[str]:
    """
    Extract reply list from LLM output with four fallback layers.

    Preprocessing (before any layer):
      Strip <think>...</think> blocks — qwen3 thinking output.

    Layer 1: clean JSON → {"replies": ["a", "b", "c"]}
    Layer 2: JSON fenced in ```...```
    Layer 3: JSON object embedded in prose
    Layer 4: numbered list ("1. reply text")

    Never raises — always returns a (possibly empty) list.
    """
    if not raw:
        return []

    # ── Pre-process: strip thinking blocks ────────────────────────────────────
    text = _strip_thinking(raw)
    if not text:
        return []

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # Layer 1: direct JSON parse
    try:
        data    = json.loads(text)
        replies = _extract_replies_from_json(data)
        if replies:
            return replies[:expected]
    except json.JSONDecodeError:
        pass

    # Layer 2: JSON object embedded in prose
    #   Note: now safe to use [^{}]* because thinking blocks are already stripped
    match = re.search(
        r'\{[^{}]*"(?:replies|options|drafts)"\s*:\s*\[.*?\]\s*\}',
        text,
        re.DOTALL,
    )
    if match:
        try:
            replies = _extract_replies_from_json(json.loads(match.group()))
            if replies:
                return replies[:expected]
        except json.JSONDecodeError:
            pass

    # Layer 3: numbered list fallback
    numbered = re.findall(r'^\d+\.\s+(.+)$', text, re.MULTILINE)
    if numbered:
        return [n.strip().strip('"') for n in numbered[:expected]]

    # Layer 4: quoted strings (last resort)
    quoted = [
        q for q in re.findall(r'"([^"]{4,200})"', text)
        if not _is_json_key(q)
    ]
    if quoted:
        return quoted[:expected]

    return []


def _extract_replies_from_json(data) -> list[str]:
    """Extract reply strings from a parsed JSON dict or list."""
    if isinstance(data, dict):
        for key in ("replies", "options", "drafts"):
            if key in data and isinstance(data[key], list):
                return [str(r).strip() for r in data[key] if str(r).strip()]
    if isinstance(data, list):
        return [str(r).strip() for r in data if str(r).strip()]
    return []


def _is_json_key(s: str) -> bool:
    return s in {"replies", "options", "drafts", "reply", "option", "draft"}


# ── Helpers ────────────────────────────────────────────────────────────────────

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