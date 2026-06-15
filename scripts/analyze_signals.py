#!/usr/bin/env python
"""
scripts/analyze_signals.py

Relationship signal analysis for a chat — terminal version of the
`python -m cli.main signals` command, for quick local debugging
without going through the full CLI/Rich formatting.

v2 changes:
    This script previously duplicated all vocabulary sets (SLANG_WORDS,
    HINDI_WORDS, etc.) AND reimplemented scoring logic separately from
    classifier.py. The two diverged — this script could show a different
    "most likely" relationship type than the real classifier used by
    `setup`, which is misleading.

    Now this script imports directly from brain.profiler.classifier and
    calls get_all_scores() — the SAME function the CLI `signals` command
    uses. This script is now purely a display/formatting wrapper. There
    is exactly one source of truth for relationship scoring.

Usage:
    python scripts/analyze_signals.py "data/exports/WhatsApp Chat with Shreyansh.txt" --you "Rishit Babbar"
    python scripts/analyze_signals.py data/exports/ --you "Rishit Babbar"   # all files
"""
import sys
import argparse
from pathlib import Path

# ── CRITICAL: Add project root to sys.path ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain.profiler.parsers.whatsapp import parse_whatsapp_export
from brain.profiler.parsers.telegram import parse_telegram_export
from brain.profiler.analyzer import _group_by_contact
from brain.profiler.classifier import get_all_scores


TYPE_LABELS = {
    "close_friend":      "🔥 CLOSE FRIEND",
    "academic_peer":     "📚 ACADEMIC PEER",
    "family":            "👨‍👩‍👧‍👦 FAMILY",
    "colleague":         "💼 COLLEAGUE",
    "romantic_interest": "💕 ROMANTIC INTEREST",
    "acquaintance":      "🤝 ACQUAINTANCE",
}


def _parse_file(file_path: Path):
    """Parse a single export file. Returns list of MessageRecord."""
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return parse_whatsapp_export(file_path)
    elif ext == ".json":
        return parse_telegram_export(file_path)
    else:
        print(f"❌ Unknown format: {ext}")
        return []


def analyze_signals(file_path: Path, you: str):
    """Show relationship signal breakdown for every contact in a chat export."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"📖 Parsing {file_path.name}...")
    try:
        records = _parse_file(file_path)
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return

    if not records:
        print("❌ No messages parsed")
        return

    print(f"✓ Parsed {len(records)} messages\n")

    contact_groups = _group_by_contact(records, you)

    for contact_name, contact_records in contact_groups.items():
        user_msgs = [r for r in contact_records if r.sender == you]
        if len(user_msgs) < 5:
            continue

        print("=" * 80)
        print(f" {contact_name}  —  {len(user_msgs)} messages from {you}".center(80))
        print("=" * 80)

        # ── This is the ONE source of truth — same function the CLI uses ────────
        all_scores = get_all_scores(user_msgs, contact_records)

        if not all_scores:
            print("\n  Insufficient data for classification (need ≥10 messages)\n")
            continue

        for rel_type, (confidence, signals) in all_scores.items():
            label = TYPE_LABELS.get(rel_type, rel_type.upper())
            bar = "█" * int(confidence * 20)
            print(f"\n{label:<28} {confidence:>5.0%}  {bar}")
            if signals:
                for sig in signals:
                    print(f"    → {sig}")
            else:
                print(f"    → no signals")

        best_type, best_conf = max(
            ((t, c) for t, (c, _) in all_scores.items()),
            key=lambda x: x[1],
        )
        print(f"\n{'─' * 80}")
        print(f"  → Classified as: {TYPE_LABELS.get(best_type, best_type)} ({best_conf:.0%} confidence)")
        print()


def analyze_directory(directory: Path, you: str):
    """Analyze all chats in a directory."""
    directory = Path(directory)

    if not directory.is_dir():
        print(f"❌ Not a directory: {directory}")
        return

    files = list(directory.glob("*.txt")) + list(directory.glob("*.json"))

    if not files:
        print(f"❌ No chat files found in {directory}")
        return

    for file_path in sorted(files):
        analyze_signals(file_path, you)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show relationship signal breakdown for chat exports."
    )
    parser.add_argument("path", help="Chat export file OR directory")
    parser.add_argument(
        "--you", "-u", required=True,
        help="Your name as it appears in the exports",
    )
    args = parser.parse_args()

    path_obj = Path(args.path)

    if path_obj.is_dir():
        analyze_directory(path_obj, args.you)
    else:
        analyze_signals(path_obj, args.you)