"""
brain/profiler/models.py

Data contracts for PersonaOS.

Every piece of data that flows through the system is defined here.
These are the source of truth for what a message, a style profile,
and a communication profile look like.

Why Pydantic?
    Pydantic enforces types at runtime. If a timestamp is not a valid
    datetime, Pydantic raises immediately — not 500 lines later when
    something silently computes a wrong number. For a data pipeline
    processing thousands of messages, this is essential.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class MessageRecord(BaseModel):
    """
    A single parsed message from any chat export.

    This is the atomic unit of the PersonaOS data pipeline.
    Every parser (WhatsApp, Telegram, iMessage) produces these.
    Every analyzer consumes these.
    """
    timestamp: datetime
    sender: str
    content: str
    platform: str              # "whatsapp" | "telegram" | "imessage"
    chat_name: str             # group name or contact name
    has_emoji: bool = False
    has_media: bool = False    # photos, videos, stickers — content will be ""
    word_count: int = 0
    char_count: int = 0


class EmojiProfile(BaseModel):
    """Emoji usage patterns for a user."""
    total_emoji_count: int = 0
    unique_emoji_count: int = 0
    top_emojis: list[tuple[str, int]] = []    # [("😂", 45), ("🔥", 20)]
    emoji_frequency: float = 0.0              # emojis per message


class StyleProfile(BaseModel):
    """
    How a user writes — tone, length, structure.

    This is the core output of the analyzer. It captures the
    measurable, statistical properties of someone's writing style.
    Used by the generator as the foundation of every prompt.
    """
    avg_message_length: float = 0.0           # characters
    avg_word_count: float = 0.0
    uses_lowercase_heavily: bool = False       # >60% messages start lowercase
    uses_punctuation: bool = True             # >30% messages end with . ! ? ,
    common_openers: list[str] = []            # ["hey", "yo", "bro"]
    common_closers: list[str] = []            # ["lol", "haha", "ok"]
    slang_tokens: list[str] = []             # slang words the user actually uses
    typing_quirks: list[str] = []            # ["no caps", "trailing dots...", "burst texter"]


class RelationshipProfile(BaseModel):
    """
    How the user communicates with a specific person.

    The same user writes differently to different contacts.
    This captures that relationship-specific style.
    """
    contact_name: str
    message_count: int = 0
    style: Optional[StyleProfile] = None
    emoji_usage: Optional[EmojiProfile] = None
    tone_notes: list[str] = []               # ["emoji-heavy", "asks lots of questions"]


class CommunicationProfile(BaseModel):
    """
    Top-level profile for a user. This is the main output of Phase 1.

    Contains:
    - Global style (how they write in general)
    - Global emoji patterns
    - Per-contact relationship profiles
    - A raw LLM-generated summary (Phase 2)
    """
    created_at: datetime = Field(default_factory=datetime.now)
    platform: str
    total_messages_analyzed: int = 0
    global_style: Optional[StyleProfile] = None
    global_emoji: Optional[EmojiProfile] = None
    relationships: list[RelationshipProfile] = []
    raw_summary: str = ""                    # LLM-generated summary (Phase 2)