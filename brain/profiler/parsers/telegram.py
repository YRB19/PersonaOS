"""
brain/profiler/parsers/telegram.py

Parses Telegram chat exports (JSON format).

How to export from Telegram:
  Desktop app → Settings → Advanced → Export Telegram Data
  Select: Personal chats, Group chats
  Format: Machine-readable JSON
  This produces a 'result.json' file.

Telegram JSON structure:
  {
    "name": "Chat Name",
    "type": "personal_chat" | "private_group" | "private_supergroup",
    "messages": [
      {
        "id": 1,
        "type": "message",
        "date": "2024-01-15T14:30:00",
        "from": "Rishit Sharma",
        "from_id": "user123456",
        "text": "hey what's up",
        "reactions": [                         ← Phase 3A addition
          {
            "type": "emoji",
            "emoji": "👍",
            "count": 2,
            "chosen": true     ← true = YOU reacted with this emoji
          }
        ]
      }
    ]
  }

Phase 3A — reaction parsing:
  Telegram exports include a `reactions` field on each message.
  `"chosen": true` marks reactions that the EXPORTING USER sent.
  We extract only chosen reactions and store them in MessageRecord.reactions.
  These are aggregated by the analyzer into EmojiProfile.top_reactions.

  WhatsApp note: WhatsApp .txt exports do NOT include reactions at all.
  This is a hard platform limitation, not a code gap.
"""
import json
from datetime import datetime
from pathlib import Path
from brain.profiler.models import MessageRecord


# Hard size limits to bound pathological inputs.
# Telegram exports are typically a few MB; these caps reject inputs that
# would otherwise force the parser to build multi-GB in-memory structures.
MAX_EXPORT_BYTES  = 256 * 1024 * 1024   # 256 MB hard cap on the file
MAX_MESSAGES      = 2_000_000           # hard cap on parsed messages
MAX_TEXT_ENTITIES = 10_000             # per-message entity list cap
MAX_TEXT_CHARS    = 64_000             # per-message reconstructed text cap

# Message types to skip — service events, not actual messages
SKIP_TYPES = {
    "service",
    "unsupported",
}

# Text entity types that add no readable content
SKIP_ENTITY_TYPES = {
    "mention_name",
    "phone",
    "cashtag",
}


def parse_telegram_export(
    file_path: str | Path,
    chat_name: str = "",
) -> list[MessageRecord]:
    """
    Parse a Telegram JSON export file.

    Args:
        file_path: Path to result.json (or renamed .json file)
        chat_name: Override chat name (defaults to name from JSON)

    Returns:
        List of MessageRecord objects, sorted by timestamp.
        MessageRecord.reactions populated where the user sent reactions.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")

    size = path.stat().st_size
    if size > MAX_EXPORT_BYTES:
        raise ValueError(
            f"Telegram export too large to parse safely: {size / 1e6:.0f} MB "
            f"(limit {MAX_EXPORT_BYTES / 1e6:.0f} MB)."
        )

    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError(
            f"Unexpected JSON structure in {path}. "
            "Make sure you exported in 'Machine-readable JSON' format."
        )

    if not isinstance(data["messages"], list):
        raise ValueError(f"'messages' must be a list in {path}.")

    resolved_chat_name = chat_name or data.get("name", path.stem)
    records: list[MessageRecord] = []

    for msg in data["messages"]:
        if len(records) >= MAX_MESSAGES:
            break
        if not isinstance(msg, dict):
            continue
        record = _parse_message(msg, resolved_chat_name)
        if record:
            records.append(record)

    return sorted(records, key=lambda r: r.timestamp)


def parse_telegram_export_multi(
    file_paths: list[str | Path],
) -> list[MessageRecord]:
    """
    Parse multiple Telegram export files and merge them.
    Useful when you've exported multiple chats separately.

    Returns:
        All records from all files, sorted by timestamp
    """
    all_records: list[MessageRecord] = []
    for path in file_paths:
        try:
            all_records.extend(parse_telegram_export(path))
        except Exception as e:
            print(f"Warning: could not parse {path}: {e}")
    return sorted(all_records, key=lambda r: r.timestamp)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _parse_message(msg: dict, chat_name: str) -> MessageRecord | None:
    """Convert a single Telegram message dict into a MessageRecord."""

    # Skip non-message types
    if msg.get("type") in SKIP_TYPES:
        return None

    # Must have a sender
    sender = msg.get("from") or msg.get("actor")
    if not sender:
        return None

    # Parse timestamp
    timestamp = _parse_timestamp(msg.get("date", ""))
    if not timestamp:
        return None

    # Extract text content
    content = _extract_text(msg)
    has_media = _has_media(msg)

    # If no text AND no media, skip
    if not content and not has_media:
        return None

    # Phase 3A: extract emoji reactions the user gave to this message
    reactions = _extract_reactions(msg)

    return MessageRecord(
        timestamp=timestamp,
        sender=str(sender),
        content=content,
        platform="telegram",
        chat_name=chat_name,
        has_emoji=_has_emoji(content),
        has_media=has_media,
        word_count=len(content.split()) if content else 0,
        char_count=len(content),
        reactions=reactions,
    )


def _extract_reactions(msg: dict) -> list[str]:
    """
    Extract emoji reactions the current user (exporter) gave to this message.

    Telegram exports include a `reactions` array on each message.
    Each entry has:
      - "emoji":  the emoji character
      - "count":  total number of users who used this reaction
      - "chosen": true if the EXPORTING USER sent this reaction

    We return only reactions with chosen=True — these are the user's own
    reactions and are the signal we want for personalisation.

    Returns:
        List of emoji strings, e.g. ["👍", "❤️"].
        Empty list if no reactions or the field is absent.
    """
    raw = msg.get("reactions", [])
    if not raw or not isinstance(raw, list):
        return []

    result: list[str] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        emoji = r.get("emoji", "")
        chosen = r.get("chosen", False)
        if emoji and chosen:
            result.append(emoji)

    return result


def _extract_text(msg: dict) -> str:
    """
    Extract plain text from a Telegram message.

    Telegram text can be:
    1. A plain string: {"text": "hello"}
    2. A list of entities: {"text": [{"type": "plain", "text": "hello"}, ...]}
    3. Absent (for media-only messages)

    Bounded: reconstruction iterates at most MAX_TEXT_ENTITIES items
    and the resulting string is truncated at MAX_TEXT_CHARS (DoS guard).
    """
    raw = msg.get("text", "")

    if isinstance(raw, str):
        return raw[:MAX_TEXT_CHARS].strip()

    if isinstance(raw, list):
        parts: list[str] = []
        total = 0
        for i, entity in enumerate(raw):
            if i >= MAX_TEXT_ENTITIES:
                break
            if isinstance(entity, str):
                piece = entity
            elif isinstance(entity, dict):
                entity_type = entity.get("type", "plain")
                if entity_type in SKIP_ENTITY_TYPES:
                    continue
                piece = entity.get("text", "")
            else:
                continue
            if total + len(piece) > MAX_TEXT_CHARS:
                piece = piece[: MAX_TEXT_CHARS - total]
            parts.append(piece)
            total += len(piece)
            if total >= MAX_TEXT_CHARS:
                break
        return "".join(parts).strip()

    return ""


def _has_media(msg: dict) -> bool:
    """Check if the message contains media (photo, video, sticker, etc.)."""
    media_fields = {
        "photo", "video", "video_file", "voice_message",
        "audio_file", "sticker", "animation", "file",
        "location_information", "poll", "contact_information",
    }
    return any(msg.get(field) for field in media_fields)


def _parse_timestamp(date_str: str) -> datetime | None:
    """Parse Telegram's ISO 8601 datetime string."""
    if not date_str:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _has_emoji(text: str) -> bool:
    """Quick emoji detection (mirrors whatsapp.py)."""
    emoji_ranges = [
        (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF),
        (0x1F700, 0x1F77F), (0x2600, 0x26FF), (0x2700, 0x27BF),
        (0xFE00, 0xFE0F), (0x1F900, 0x1F9FF), (0x1FA00, 0x1FA6F),
        (0x1FA70, 0x1FAFF),
    ]
    for char in text:
        cp = ord(char)
        if any(s <= cp <= e for s, e in emoji_ranges):
            return True
    return False