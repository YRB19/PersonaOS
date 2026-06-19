"""
brain/profiler/models.py

Data contracts for PersonaOS.

v2 changes:
    RelationshipProfile gains relationship_type + relationship_signals
    CommunicationProfile gains sources_analyzed + total_chats_analyzed
    All new fields have defaults — backward compatible with v1 profiles.

Why Pydantic?
    Pydantic enforces types at runtime. If a timestamp is not a valid
    datetime, Pydantic raises immediately — not 500 lines later when
    something silently computes a wrong number.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Relationship type constants ────────────────────────────────────────────────

RELATIONSHIP_TYPES = {
    "close_friend",
    "college_friend",     
    "academic_peer",
    "family",
    "colleague",
    "romantic_interest",
    "acquaintance",
    "unknown",
}


class MessageRecord(BaseModel):
    """
    A single parsed message from any chat export.
    Atomic unit of the PersonaOS pipeline.
    """
    timestamp: datetime
    sender: str
    content: str
    platform: str              # "whatsapp" | "telegram" | "imessage"
    chat_name: str             # group name or contact name
    has_emoji: bool = False
    has_media: bool = False
    word_count: int = 0
    char_count: int = 0


class EmojiProfile(BaseModel):
    """Emoji usage patterns for a user."""
    total_emoji_count: int = 0
    unique_emoji_count: int = 0
    top_emojis: list[tuple[str, int]] = []
    emoji_frequency: float = 0.0


class StyleProfile(BaseModel):
    """
    How a user writes — tone, length, structure.
    Core output of the analyzer, used as the LLM prompt foundation.
    """
    avg_message_length: float = 0.0
    avg_word_count: float = 0.0
    uses_lowercase_heavily: bool = False
    uses_punctuation: bool = True
    common_openers: list[str] = []
    common_closers: list[str] = []
    slang_tokens: list[str] = []
    typing_quirks: list[str] = []


class RelationshipProfile(BaseModel):
    """
    How the user communicates with a specific person, and what kind
    of relationship it is.

    relationship_type values:
        "close_friend"       casual, slang-heavy, personal topics
        "academic_peer"      academic context (assignments, exams, notes)
        "family"             family address terms detected
        "colleague"          professional/work context
        "romantic_interest"  intimate/romantic signals detected
        "acquaintance"       low message count, surface-level, minimal slang
        "unknown"            insufficient signal to classify

    relationship_confidence:
        0.0–1.0, how confident the classifier is in this categorization

    relationship_signals:
        Keywords/patterns that drove the classification.
        Useful for debugging. Example:
            ["bro (x14)", "yaar (x8)", "assignment (x5)"]
    """
    contact_name: str
    message_count: int = 0
    style: Optional[StyleProfile] = None
    emoji_usage: Optional[EmojiProfile] = None
    tone_notes: list[str] = []
    relationship_type: str = "unknown"
    relationship_confidence: float = 0.0
    relationship_signals: list[str] = []


class CommunicationProfile(BaseModel):
    """
    Top-level profile for a user.

    Built from one or more chat exports. Each new export is merged
    into the existing profile rather than replacing it.

    sources_analyzed:
        Filenames of every chat export that contributed to this profile.
        Used to prevent re-processing the same file twice.
    """
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    platform: str
    total_messages_analyzed: int = 0
    total_chats_analyzed: int = 0
    sources_analyzed: list[str] = []
    global_style: Optional[StyleProfile] = None
    global_emoji: Optional[EmojiProfile] = None
    relationships: list[RelationshipProfile] = []
    raw_summary: str = ""

    @property
    def sources(self) -> list[str]:
        """Alias for sources_analyzed for backward compatibility."""
        return self.sources_analyzed