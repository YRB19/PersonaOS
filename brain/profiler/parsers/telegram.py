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
        "text": "hey what's up",         ← plain string
        "text_entities": [...]           ← OR structured entities
      }
    ]
  }
"""
import json
from datetime import datetime
from pathlib import Path
from brain.profiler.models import MessageRecord


# Message types to skip — service events, not actual messages
SKIP_TYPES = {
    "service",        # group join/leave, title changes
    "unsupported",    # old unsupported message types
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
        List of MessageRecord objects, sorted by timestamp
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "messages" not in data:
        raise ValueError(
            f"Unexpected JSON structure in {path}. "
            "Make sure you exported in 'Machine-readable JSON' format."
        )

    resolved_chat_name = chat_name or data.get("name", path.stem)
    records: list[MessageRecord] = []

    for msg in data["messages"]:
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
    )


def _extract_text(msg: dict) -> str:
    """
    Extract plain text from a Telegram message.

    Telegram text can be:
    1. A plain string: {"text": "hello"}
    2. A list of entities: {"text": [{"type": "plain", "text": "hello"}, ...]}
    3. Absent (for media-only messages)
    """
    raw = msg.get("text", "")

    if isinstance(raw, str):
        return raw.strip()

    if isinstance(raw, list):
        # Reconstruct from entity list
        parts: list[str] = []
        for entity in raw:
            if isinstance(entity, str):
                parts.append(entity)
            elif isinstance(entity, dict):
                entity_type = entity.get("type", "plain")
                if entity_type not in SKIP_ENTITY_TYPES:
                    parts.append(entity.get("text", ""))
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