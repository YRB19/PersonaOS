"""
WhatsApp chat export parser.

Handles the standard .txt format WhatsApp exports:
  [DD/MM/YYYY, HH:MM:SS] Sender: Message
  [MM/DD/YYYY, HH:MM:SS] Sender: Message  ← US format
"""
import re
from datetime import datetime
from pathlib import Path
from brain.profiler.models import MessageRecord


# WhatsApp uses different date formats by region
TIMESTAMP_PATTERNS = [
    # International: [DD/MM/YYYY, HH:MM:SS]
    re.compile(
        r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]\s(.+?):\s(.+)$'
    ),
    # Some exports: DD/MM/YY, HH:MM -
    re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[AP]M)?)\s-\s(.+?):\s(.+)$'
    ),
]

MEDIA_PLACEHOLDER = "<Media omitted>"
SYSTEM_MESSAGES = [
    "Messages and calls are end-to-end encrypted",
    "changed the subject",
    "added",
    "removed",
    "left",
    "created group",
    "changed this group's icon",
]


def parse_whatsapp_export(file_path: str | Path, chat_name: str = "") -> list[MessageRecord]:
    """
    Parse a WhatsApp .txt export file.
    
    Args:
        file_path: Path to the exported .txt file
        chat_name: Optional name for this chat (defaults to filename)
    
    Returns:
        List of MessageRecord objects, sorted by timestamp
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")
    
    if not chat_name:
        chat_name = path.stem

    records: list[MessageRecord] = []
    
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
        lines = f.readlines()

    current_record: dict | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parsed = _try_parse_line(line)
        
        if parsed:
            # Save previous record before starting new one
            if current_record:
                record = _build_record(current_record, chat_name)
                if record:
                    records.append(record)
            current_record = parsed
        else:
            # Continuation of previous message (multi-line message)
            if current_record:
                current_record["content"] += "\n" + line

    # Don't forget the last record
    if current_record:
        record = _build_record(current_record, chat_name)
        if record:
            records.append(record)

    return sorted(records, key=lambda r: r.timestamp)


def _try_parse_line(line: str) -> dict | None:
    """Try to match a line against known WhatsApp timestamp patterns."""
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.match(line)
        if match:
            date_str, time_str, sender, content = match.groups()
            return {
                "date_str": date_str.strip(),
                "time_str": time_str.strip(),
                "sender": sender.strip(),
                "content": content.strip(),
            }
    return None


def _build_record(raw: dict, chat_name: str) -> MessageRecord | None:
    """Convert raw parsed dict to a MessageRecord, or None if it's a system message."""
    content = raw["content"]
    
    # Skip WhatsApp system messages
    if any(sys_msg in content for sys_msg in SYSTEM_MESSAGES):
        return None
    
    # Try to parse timestamp
    timestamp = _parse_timestamp(raw["date_str"], raw["time_str"])
    if not timestamp:
        return None

    has_media = content == MEDIA_PLACEHOLDER
    if has_media:
        content = ""

    return MessageRecord(
        timestamp=timestamp,
        sender=raw["sender"],
        content=content,
        platform="whatsapp",
        chat_name=chat_name,
        has_emoji=_has_emoji(content),
        has_media=has_media,
        word_count=len(content.split()) if content else 0,
        char_count=len(content),
    )


def _parse_timestamp(date_str: str, time_str: str) -> datetime | None:
    """Try multiple date formats — WhatsApp varies by region."""
    formats = [
        "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%y %I:%M %p",
        "%d/%m/%y %H:%M",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%y %H:%M",
    ]
    combined = f"{date_str} {time_str}"
    for fmt in formats:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def _has_emoji(text: str) -> bool:
    """Quick emoji detection without importing emoji lib for performance."""
    # Unicode ranges for common emoji blocks
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc symbols
        (0x1F680, 0x1F6FF),  # Transport
        (0x1F700, 0x1F77F),  # Alchemical
        (0x2600, 0x26FF),    # Misc symbols
        (0x2700, 0x27BF),    # Dingbats
        (0xFE00, 0xFE0F),    # Variation selectors
        (0x1F900, 0x1F9FF),  # Supplemental
        (0x1FA00, 0x1FA6F),  # Extended-A
        (0x1FA70, 0x1FAFF),  # Extended-B
    ]
    for char in text:
        cp = ord(char)
        if any(start <= cp <= end for start, end in emoji_ranges):
            return True
    return False