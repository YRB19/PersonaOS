"""
brain/profiler/models.py

Data contracts for PersonaOS.

v4 changes (reply-system-v6):
  FeedbackSummary   — stats model for the user feedback loop.
                      Computed on read from the feedback JSONL store,
                      never persisted. Reports acceptance rate and
                      top patterns.

v3 changes (Phase 3A):
  MessageRecord:  +reactions field (list[str]) — emoji reactions the user gave
                  to THIS message. Populated by Telegram parser only.
                  Always [] for WhatsApp (platform limitation — reactions
                  are not included in WhatsApp .txt exports).

  EmojiProfile:   +reaction_count, +top_reactions — aggregated from
                  reactions across all messages in the conversation.
                  Both default to 0/[] for backward compatibility.

v2 changes:
    RelationshipProfile gains relationship_type + relationship_signals
    CommunicationProfile gains sources_analyzed + total_chats_analyzed
    All new fields have defaults — backward compatible with v1 profiles.
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

    reactions:
        Emojis the profiled user sent as reactions TO this message.
        Populated only for Telegram exports (Telegram JSON includes
        reaction data; WhatsApp .txt exports do not).
        Example: if the user reacted 👍 to a message from Priya,
        that message's reactions list will contain ["👍"].
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
    reactions: list[str] = Field(default_factory=list)


class EmojiProfile(BaseModel):
    """
    Emoji usage patterns for a user.

    top_emojis:    Emojis the user sends IN text messages (Phase 1 data)
    top_reactions: Emojis the user sends AS REACTIONS (Phase 3 / Telegram only)

    These are distinct signals:
    - top_emojis tells you how the user expresses themselves in messages
    - top_reactions tells you how they respond to OTHER people's messages

    Both are used by emoji_suggester.py to personalise reaction suggestions.
    """
    total_emoji_count: int = 0
    unique_emoji_count: int = 0
    top_emojis: list[tuple[str, int]] = []
    emoji_frequency: float = 0.0
    # Phase 3 additions — reaction data (Telegram exports only)
    reaction_count: int = 0
    top_reactions: list[tuple[str, int]] = []


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
        "college_friend"     close peer + academic collaboration
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


class FeedbackSummary(BaseModel):
    """
    Stats about user feedback on generated reply drafts.

    Computed on read from the feedback JSONL store — never persisted.
    Reports acceptance rate and top patterns.
    """
    total_recorded: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    acceptance_rate: float = 0.0
    top_accepted_patterns: list[str] = []


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