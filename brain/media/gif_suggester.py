"""
brain/media/gif_suggester.py

GIF search via Giphy API (free tier — 100 requests/hour).

Migration note (2026-06-30):
  Google permanently shut down the Tenor API on June 30, 2026.
  All Tenor keys stopped working. Giphy is the standard replacement —
  used by Slack, Discord, iMessage, and most major platforms.

Setup (one-time, ~2 minutes):
  1. https://developers.giphy.com/dashboard/
  2. Create an app (select "API") → copy the API key
  3. Add to .env: GIPHY_API_KEY=your_key_here

Without an API key: returns [] silently.
Emoji suggestions continue to work fully with no key.
"""
import os
import re
from typing import Optional
from urllib.parse import urlparse

import requests

from brain.media.models import GifResult
from brain.profiler.models import RelationshipProfile


GIPHY_BASE      = "https://api.giphy.com/v1/gifs/search"
DEFAULT_LIMIT   = 5
REQUEST_TIMEOUT = 5

ALLOWED_GIF_SCHEMES = {"https", "http"}

STOP_WORDS = {
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

RELATIONSHIP_TONE: dict[str, str] = {
    "close_friend":      "friends reaction",
    "college_friend":    "college friends",
    "family":            "family wholesome",
    "romantic_interest": "sweet cute",
    "colleague":         "professional reaction",
    "acquaintance":      "reaction",
    "unknown":           "",
}


def search_gifs(
    query: str,
    relationship: Optional[RelationshipProfile] = None,
    limit: int = DEFAULT_LIMIT,
) -> list[GifResult]:
    """
    Search Giphy for GIFs relevant to the given context.
    Returns [] if no GIPHY_API_KEY in env or request fails.
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
                "rating":  "pg-13",
                "lang":    "en",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_giphy_response(resp.json())
    except Exception:
        return []


def build_gif_query(
    text: str,
    relationship: Optional[RelationshipProfile] = None,
) -> str:
    """
    Build a clean Giphy search query from raw message text.
    Public so it can be tested without an API key.
    """
    words    = re.findall(r'[a-zA-Z]{3,}', text.lower())
    keywords = [w for w in words if w not in STOP_WORDS][:3]

    tone_mod = ""
    if relationship:
        tone_mod = RELATIONSHIP_TONE.get(relationship.relationship_type, "")

    parts = keywords + ([tone_mod] if tone_mod else [])
    query = " ".join(parts).strip()
    return query if query else "reaction"


def _parse_giphy_response(data: dict) -> list[GifResult]:
    """
    Parse Giphy v1 API response.
    Only http(s) URLs accepted — rejects javascript:, file:, data: schemes.
    """
    results: list[GifResult] = []

    for item in data.get("data", []):
        if not isinstance(item, dict):
            continue
        images = item.get("images", {})
        if not isinstance(images, dict):
            continue

        original    = images.get("original", {}) or {}
        preview     = images.get("fixed_height_small", images.get("preview_gif", {})) or {}
        url         = original.get("url", "") if isinstance(original, dict) else ""
        preview_url = preview.get("url", url) if isinstance(preview, dict) else url
        url         = url if isinstance(url, str) else ""
        preview_url = preview_url if isinstance(preview_url, str) else ""

        if not _is_safe_gif_url(url):
            continue
        if preview_url and not _is_safe_gif_url(preview_url):
            preview_url = url

        results.append(GifResult(
            title=item.get("title", "GIF") if isinstance(item.get("title"), str) else "GIF",
            preview_url=preview_url or url,
            url=url,
            source="giphy",
        ))

    return results


def _is_safe_gif_url(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return False
    return parsed.scheme in ALLOWED_GIF_SCHEMES and bool(parsed.netloc)
