"""
WhatsApp chat export parser.

Handles the standard .txt format WhatsApp exports.

FORMAT VARIANTS (both supported):
  Bracketed   — [DD/MM/YYYY, HH:MM:SS] Sender: Message
  Dash        — DD/MM/YYYY, H:MM am/pm - Sender: Message  ← common on Android/India

KEY FIX (v2):
  Real WhatsApp exports from Android and Indian accounts use lowercase am/pm.
  Pattern 2 now compiled with re.IGNORECASE to handle both 'pm' and 'PM'.
  _parse_timestamp normalises the combined string to uppercase before strptime
  so %p matches reliably across all macOS/Linux locales.
"""
import re
from datetime import datetime
from pathlib import Path
from brain.profiler.models import MessageRecord


# ── Timestamp patterns ─────────────────────────────────────────────────────────
# Both patterns compiled with IGNORECASE so am/pm/AM/PM all match.

TIMESTAMP_PATTERNS = [
    # Bracketed format: [DD/MM/YYYY, HH:MM:SS] Sender: Message
    # Common on iOS exports and older Android versions.
    re.compile(
        r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s'
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]\s'
        r'(.+?):\s(.+)$',
        re.IGNORECASE,
    ),
    # Dash format: DD/MM/YYYY, H:MM am/pm - Sender: Message
    # Common on Android exports, Indian locale accounts.
    # CRITICAL: [AP]M alone misses lowercase 'am'/'pm' — use IGNORECASE.
    re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s'
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s-\s'
        r'(.+?):\s?(.*)$',   # (.*)  ← allows empty content (reactions, deleted msgs)
        re.IGNORECASE,
    ),
]

MEDIA_PLACEHOLDER = "<Media omitted>"

# Substrings that identify WhatsApp system/service messages (not real messages).
# Using substring match so these work even when WhatsApp appends extra text.
SYSTEM_MESSAGE_FRAGMENTS = [
    "Messages and calls are end-to-end encrypted",
    "changed the subject",
    " added ",
    " removed ",
    " left",
    "created group",
    "changed this group",
    "pinned a message",
    "changed their phone number",
    "security code changed",
    "disappearing messages",
]

# Suffixes WhatsApp appends that pollute message content
CONTENT_SUFFIXES_TO_STRIP = [
    " <This message was edited>",
    "<This message was edited>",
]

# File attachment patterns — WhatsApp shows these as text when exported without media
ATTACHMENT_PATTERNS = [
    re.compile(r'\.(vcf|pdf|docx?|xlsx?|pptx?|zip|mp3|mp4|mov|avi)\s*\(file attached\)$', re.IGNORECASE),
]


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_whatsapp_export(
    file_path: str | Path,
    chat_name: str = "",
) -> list[MessageRecord]:
    """
    Parse a WhatsApp .txt export file.

    Args:
        file_path: Path to the exported .txt file
        chat_name: Optional name for this chat (defaults to filename stem)

    Returns:
        List of MessageRecord objects sorted by timestamp.
        Returns empty list (not an error) if no parseable messages found.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")

    if not chat_name:
        chat_name = path.stem

    records: list[MessageRecord] = []

    # utf-8-sig strips the BOM that WhatsApp adds on some devices
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    current_record: dict | None = None

    for line in lines:
        line = line.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue

        parsed = _try_parse_line(line)

        if parsed:
            # Flush previous record before starting a new one
            if current_record:
                record = _build_record(current_record, chat_name)
                if record:
                    records.append(record)
            current_record = parsed
        else:
            # Multi-line message continuation
            if current_record:
                current_record["content"] += "\n" + line.strip()

    # Flush the last record
    if current_record:
        record = _build_record(current_record, chat_name)
        if record:
            records.append(record)

    return sorted(records, key=lambda r: r.timestamp)


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _try_parse_line(line: str) -> dict | None:
    """
    Try to match a line against known WhatsApp timestamp patterns.
    Returns a raw dict if matched, None if this line is a continuation.
    """
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.match(line)
        if match:
            date_str, time_str, sender, content = match.groups()
            return {
                "date_str": date_str.strip(),
                "time_str": time_str.strip(),
                "sender": sender.strip(),
                "content": content.strip() if content else "",
            }
    return None


def _build_record(raw: dict, chat_name: str) -> MessageRecord | None:
    """
    Convert a parsed raw dict into a MessageRecord.
    Returns None for system messages and messages that fail timestamp parsing.
    """
    content = raw["content"]

    # Skip WhatsApp system/service messages
    if _is_system_message(content):
        return None

    # Parse timestamp
    timestamp = _parse_timestamp(raw["date_str"], raw["time_str"])
    if not timestamp:
        return None

    # Strip known noise suffixes
    content = _clean_content(content)

    # Classify media
    has_media = (content == MEDIA_PLACEHOLDER) or _is_file_attachment(content)
    if content == MEDIA_PLACEHOLDER:
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


def _is_system_message(content: str) -> bool:
    """Return True if this content looks like a WhatsApp system notification."""
    return any(fragment in content for fragment in SYSTEM_MESSAGE_FRAGMENTS)


def _is_file_attachment(content: str) -> bool:
    """Return True if the message is a file attachment placeholder."""
    return any(p.search(content) for p in ATTACHMENT_PATTERNS)


def _clean_content(content: str) -> str:
    """Strip WhatsApp-appended suffixes that pollute message content."""
    for suffix in CONTENT_SUFFIXES_TO_STRIP:
        if content.endswith(suffix):
            content = content[: -len(suffix)].strip()
    return content


def _parse_timestamp(date_str: str, time_str: str) -> datetime | None:
    """
    Parse timestamp from the two extracted string groups.

    IMPORTANT: Normalise to uppercase before strptime.
    Python's %p directive is locale-dependent — on some macOS/Linux
    configurations it only accepts 'AM'/'PM', not 'am'/'pm'.
    Calling .upper() makes this locale-safe.
    """
    # Normalise: '3:02 pm' → '3:02 PM', '15:30' stays '15:30'
    combined = f"{date_str} {time_str}".upper()

    formats = [
        # 24-hour with seconds
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%y %H:%M:%S",
        # 24-hour without seconds
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
        "%d/%m/%y %H:%M",
        # 12-hour with seconds
        "%d/%m/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%y %I:%M:%S %p",
        # 12-hour without seconds  ← covers '25/02/2026 3:02 PM'
        "%d/%m/%Y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%d/%m/%y %I:%M %p",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def _has_emoji(text: str) -> bool:
    """Quick emoji detection using Unicode block ranges."""
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc symbols and pictographs
        (0x1F680, 0x1F6FF),  # Transport and map symbols
        (0x1F700, 0x1F77F),  # Alchemical symbols
        (0x2600,  0x26FF),   # Misc symbols
        (0x2700,  0x27BF),   # Dingbats
        (0xFE00,  0xFE0F),   # Variation selectors
        (0x1F900, 0x1F9FF),  # Supplemental symbols
        (0x1FA00, 0x1FA6F),  # Chess/extended
        (0x1FA70, 0x1FAFF),  # Symbols extended-B
    ]
    for char in text:
        cp = ord(char)
        if any(start <= cp <= end for start, end in emoji_ranges):
            return True
    return False