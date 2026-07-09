"""
brain/generator/feedback.py

Stores user feedback on generated replies for adaptive learning.
No fine-tuning — just high-weight RAG examples.

Feedback entries stored as JSONL in data/feedback/{user_name}.jsonl —
one JSON object per line, append-only.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FeedbackEntry:
    incoming_message: str
    chosen_reply: str
    rejected_replies: list[str]
    contact_name: str
    user_name: str
    timestamp: str
    relationship_type: str = "unknown"


class FeedbackStore:

    def __init__(self, path: str | Path = "data/feedback"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _user_path(self, user_name: str) -> Path:
        safe = user_name.replace(" ", "_").lower()
        return self.path / f"{safe}.jsonl"

    def record(
        self,
        incoming: str,
        chosen: str,
        rejected: list[str],
        contact: str,
        user: str,
        rel_type: str = "unknown",
    ) -> FeedbackEntry:
        entry = FeedbackEntry(
            incoming_message=incoming,
            chosen_reply=chosen,
            rejected_replies=rejected,
            contact_name=contact,
            user_name=user,
            timestamp=datetime.now(timezone.utc).isoformat(),
            relationship_type=rel_type,
        )
        data = {
            "incoming_message": entry.incoming_message,
            "chosen_reply": entry.chosen_reply,
            "rejected_replies": entry.rejected_replies,
            "contact_name": entry.contact_name,
            "user_name": entry.user_name,
            "timestamp": entry.timestamp,
            "relationship_type": entry.relationship_type,
        }
        filepath = self._user_path(user)
        with open(filepath, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return entry

    def _read_entries(self, user_name: str) -> list[dict]:
        filepath = self._user_path(user_name)
        if not filepath.exists():
            return []
        entries: list[dict] = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def get_positive_examples(
        self, user_name: str, contact: str | None = None, limit: int = 10
    ) -> list[dict]:
        entries = self._read_entries(user_name)
        if contact:
            entries = [e for e in entries if e.get("contact_name") == contact]
        chosen = [
            {
                "content": e["chosen_reply"],
                "incoming": e["incoming_message"],
                "contact_name": e["contact_name"],
                "timestamp": e["timestamp"],
                "source": "feedback_positive",
            }
            for e in entries
        ]
        return chosen[-limit:][::-1]

    def get_negative_examples(
        self, user_name: str, contact: str | None = None, limit: int = 10
    ) -> list[dict]:
        entries = self._read_entries(user_name)
        if contact:
            entries = [e for e in entries if e.get("contact_name") == contact]
        rejected: list[dict] = []
        for e in entries:
            for r in e.get("rejected_replies", []):
                rejected.append({
                    "content": r,
                    "incoming": e["incoming_message"],
                    "contact_name": e["contact_name"],
                    "timestamp": e["timestamp"],
                    "source": "feedback_negative",
                })
        return rejected[-limit:][::-1]

    def stats(self, user_name: str) -> dict:
        entries = self._read_entries(user_name)
        total = len(entries)
        accepted = total
        rejected_count = sum(len(e.get("rejected_replies", [])) for e in entries)
        acceptance_rate = 1.0 if total == 0 else accepted / max(total + rejected_count, 1)
        top_patterns: list[str] = []
        if entries:
            recent = entries[-10:]
            for e in recent:
                chosen = e.get("chosen_reply", "")
                if chosen and len(chosen) > 5:
                    top_patterns.append(chosen[:80])
        return {
            "total_recorded": total,
            "accepted_count": accepted,
            "rejected_count": rejected_count,
            "acceptance_rate": round(acceptance_rate, 4),
            "top_accepted_patterns": top_patterns,
        }