"""
brain/media/models.py

Data contracts for Phase 3 — Media Intelligence.

Three suggestion types:
  ReactionSuggestion  — emoji to react to an incoming message with
  EmojiSuggestion     — emoji to include in a text reply (different from reaction)
  GifResult           — a GIF returned from Tenor search
  MediaSuggestion     — bundled output from the full media pipeline

Design note: reaction vs reply emoji are different UX flows.
  Reaction: one-tap emoji overlay on the message (like Instagram/WhatsApp react)
  Reply emoji: emoji embedded in the reply text itself (handled by Phase 2B)
Phase 3 covers reactions primarily. Reply emoji are already handled by the
style profile in the generator prompt.
"""
from pydantic import BaseModel
from typing import Optional


class ReactionSuggestion(BaseModel):
    """
    A suggested emoji reaction to an incoming message.

    emoji:      The emoji character (e.g. "😂", "👍", "🔥")
    confidence: 0.0–1.0, how well this emoji fits the message context
    reason:     Human-readable explanation (shown in CLI, useful for debugging)
    source:     "tone_pool" | "profile" | "reaction_history"
    """
    emoji: str
    confidence: float = 0.0
    reason: str = ""
    source: str = "tone_pool"


class GifResult(BaseModel):
    """
    A single GIF from Tenor search.

    preview_url: Small/low-res preview (for displaying in a list)
    url:         Full GIF URL (the one to actually send)
    source:      Always "tenor" for now
    """
    title: str
    preview_url: str
    url: str
    source: str = "tenor"


class MediaSuggestion(BaseModel):
    """
    Bundled media suggestions for a given incoming message.

    gif_query:           The search query used (shown in CLI so user can refine it)
    reaction_suggestions: Ranked list of emoji reactions
    gif_results:          GIF results from Tenor (empty if no API key)
    """
    incoming_message: str
    from_contact: str = ""
    reaction_suggestions: list[ReactionSuggestion] = []
    gif_query: str = ""
    gif_results: list[GifResult] = []