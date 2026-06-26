"""
brain/media/gif_suggester.py

GIF search via Giphy API (free tier — 100 requests/hour).

Migration note (2026-06-30):
  Google permanently shut down the Tenor API on June 30, 2026.
  All Tenor API keys stopped working. Giphy is the standard replacement —
  used by Slack, Discord, iMessage, and most major platforms.
  The public-facing search behaviour is identical; only the endpoint
  URL and response JSON shape differ.

Setup (one-time, ~2 minutes):
  1. https://developers.giphy.com/dashboard/
  2. Create an app (select "API") → copy the API key
  3. Add to .env: GIPHY_API_KEY=your_key_here

Free tier limits:
  100 requests / hour on the public beta key.
  For a single-user local tool this is essentially unlimited.

Privacy model:
  Only a short text search query is sent to Giphy — no message content,
  no sender names, no personal identifiers. Equivalent to a Google Images
  search. Acceptable under PersonaOS privacy policy.

Without an API key:
  Returns [] silently. GIF search is opt-in.
  Emoji suggestions continue to work fully with no key.
"""
import os
import re
from typing import Optional

import requests

from brain.media.models import GifResult
from brain.profiler.models import RelationshipProfile


# ── Constants ──────────────────────────────────────────────────────────────────

GIPHY_BASE      = "https://api.giphy.com/v1/gifs/search"
DEFAULT_LIMIT   = 5
REQUEST_TIMEOUT = 5   # seconds — non-blocking for the CLI

# Stop words stripped before building a Giphy query.
# Mix of English + common Hinglish function words.
STOP_WORDS = {
    # English
    "i", "you", "he", "she", "it", "we", "they",
    "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "and", "or", "but", "in", "on",
    "at", "to", "for", "with", "from", "that", "this",
    "what", "how", "when", "where", "why", "who",
    "do", "did", "will", "would", "can", "could",
    "have", "has", "had", "just", "also", "still",
    "too", "so", "up", "out", "not", "no", "yes",
    # Hinglish function words
    "hai", "hain", "ka", "ki", "ko", "ke", "ne",
    "se", "me", "tha", "thi", "the", "kya", "aur",
    "ya", "toh", "bhi", "nahi", "nhi", "haan", "hn",
    "raha", "rahi", "rhe", "hua", "hui", "hue",
    "kar", "karo", "kr", "krna",
}

# Per-relationship tone modifiers appended to the Giphy query.
# These steer the GIF register without sending any personal context.
RELATIONSHIP_TONE: dict[str, str] = {
    "close_friend":      "friends reaction",
    "college_friend":    "college friends",
    "family":            "family wholesome",
    "romantic_interest": "sweet cute",
    "colleague":         "professional reaction",
    "acquaintance":      "reaction",
    "unknown":           "",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def search_gifs(
    query: str,
    relationship: Optional[RelationshipProfile] = None,
    limit: int = DEFAULT_LIMIT,
) -> list[GifResult]:
    """
    Search Giphy for GIFs relevant to the given context.

    Args:
        query:        The incoming message text (keywords extracted locally)
        relationship: Used to add relationship-appropriate tone modifier
        limit:        Max results (default 5)

    Returns:
        List of GifResult objects. Empty if no GIPHY_API_KEY in env,
        or if the request fails for any reason.
    """
    api_key = os.getenv("GIPHY_API_KEY", "").strip()
    if not api_key:
        return []

    search_term = build_gif_query(query, relationship)
    if not search_term:
        return []

    try:
        resp = requests.get(
            GIPHY_BASE,
            params={
                "api_key": api_key,
                "q":       search_term,
                "limit":   limit,
                "rating":  "pg-13",   # safe for general use
                "lang":    "en",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_giphy_response(resp.json())
    except Exception:
        # Any network issue, rate limit, or bad key → silent empty list
        return []


def build_gif_query(
    text: str,
    relationship: Optional[RelationshipProfile] = None,
) -> str:
    """
    Build a clean, effective search query from raw message text.

    Public so it can be tested without an API key.

    Strategy:
      1. Extract alphabetic words (strips emoji, punctuation)
      2. Filter stop words
      3. Keep top 3 meaningful keywords
      4. Append relationship tone modifier

    Examples:
      "bhai free hai kal?" + close_friend  → "free friends reaction"
      "got the internship!!" + college_friend → "internship college friends"
      "lmao that was hilarious 😂" + close_friend → "hilarious friends reaction"
    """
    words    = re.findall(r'[a-zA-Z]{3,}', text.lower())
    keywords = [w for w in words if w not in STOP_WORDS][:3]

    tone_mod = ""
    if relationship:
        tone_mod = RELATIONSHIP_TONE.get(relationship.relationship_type, "")

    parts = keywords + ([tone_mod] if tone_mod else [])
    query = " ".join(parts).strip()

    # Fallback: no extractable keywords (Hinglish-only or very short message)
    return query if query else "reaction"


# ── Internal ───────────────────────────────────────────────────────────────────

def _parse_giphy_response(data: dict) -> list[GifResult]:
    """
    Parse Giphy v1 API response into GifResult objects.

    Giphy response shape (relevant fields only):
      {
        "data": [
          {
            "title": "...",
            "images": {
              "original":          {"url": "https://media.giphy.com/.../giphy.gif"},
              "fixed_height_small": {"url": "https://media.giphy.com/.../giphy-small.gif"}
            }
          }
        ]
      }

    We use:
      url         → images.original.url          (full GIF to send)
      preview_url → images.fixed_height_small.url (small preview for listing)
    """
    results: list[GifResult] = []

    for item in data.get("data", []):
        images = item.get("images", {})

        original     = images.get("original", {})
        preview      = images.get("fixed_height_small", images.get("preview_gif", {}))

        url         = original.get("url", "")
        preview_url = preview.get("url", url)

        if url:
            results.append(GifResult(
                title=item.get("title", "GIF"),
                preview_url=preview_url,
                url=url,
                source="giphy",
            ))

    return results