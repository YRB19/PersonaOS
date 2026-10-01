"""
api/models.py

Pydantic request and response models for the PersonaOS Brain API.

These are separate from brain/profiler/models.py — they are the API
surface contract, not the internal data structures. Keeping them separate
means the internal brain models can evolve without breaking the API contract.
"""
from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class ReplyRequest(BaseModel):
    incoming: str = Field(..., max_length=2000, description="The message you received")
    from_contact: str = Field(..., max_length=100, description="Who sent it")
    user_name: str = Field(..., max_length=100, description="Your name (matches profile)")
    num_drafts: int = Field(default=3, ge=1, le=10)
    model: str = Field(default="qwen3.5:9b", max_length=100)


class ReactionRequest(BaseModel):
    incoming: str = Field(..., max_length=2000)
    user_name: str = Field(default="", max_length=100, description="Load profile for personalisation")
    contact: str = Field(default="", max_length=100)
    top_k: int = Field(default=6, ge=1, le=20)


class GifRequest(BaseModel):
    incoming: str = Field(..., max_length=2000)
    user_name: str = Field(default="", max_length=100)
    contact: str = Field(default="", max_length=100)
    limit: int = Field(default=5, ge=1, le=20)


# ── Response models ────────────────────────────────────────────────────────────

class ReplyResponse(BaseModel):
    drafts: list[str]
    model: str
    store_size: int
    contact: str
    explainability: dict | None = Field(
        default=None,
        description=(
            "Why these drafts: intent, emotion, stage, language, relationship, "
            "retrieved_examples_count/previews, has_profile, has_relationship, "
            "silence_skip, silence_reason. null only if the generator returned no record."
        ),
    )


class ReactionItem(BaseModel):
    emoji: str
    confidence: float
    reason: str
    source: str


class ReactionResponse(BaseModel):
    reactions: list[ReactionItem]
    profile_loaded: bool


class GifItem(BaseModel):
    title: str
    url: str
    preview_url: str
    source: str


class GifResponse(BaseModel):
    gifs: list[GifItem]
    query_used: str
    count: int


class ProfileRelationship(BaseModel):
    contact_name: str
    relationship_type: str
    confidence: float
    message_count: int
    top_signal: str


class ProfileResponse(BaseModel):
    name: str
    total_messages: int
    platform: str
    relationships: list[ProfileRelationship]
    avg_message_length: float
    top_emoji: list[str]
    slang_tokens: list[str]


class HealthResponse(BaseModel):
    status: str
    store_size: int
    profile_dir: str
    ollama_model: str
