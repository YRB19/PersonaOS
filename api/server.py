"""
api/server.py

Phase 3.5 — PersonaOS Brain API Server.

Exposes the brain as a local HTTP API so the Android keyboard (Phase 4)
and any other client can call it over localhost.

Architecture:
  - FastAPI with asyncio lifespan for singleton management
  - PersonaStore + ReplyGenerator loaded ONCE at startup (~5s, heavy)
  - Blocking calls (Ollama, Giphy) run in ThreadPoolExecutor so they
    don't block the event loop
  - Reaction suggestions run synchronously — pure heuristics, < 20ms

Endpoints:
  POST /suggest/reply      5-10s    Ollama + RAG
  POST /suggest/reaction   < 20ms   local heuristic, no external calls
  POST /suggest/gif        100-500ms Giphy API (requires GIPHY_API_KEY)
  GET  /profile/{name}     < 5ms    JSON read from disk
  GET  /health             < 1ms    liveness check

Run:
  python -m cli.main serve
  # or directly:
  uvicorn api.server:app --host 127.0.0.1 --port 8765 --reload
"""
import asyncio
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    GifItem, GifRequest, GifResponse,
    HealthResponse,
    ProfileRelationship, ProfileResponse,
    ReactionItem, ReactionRequest, ReactionResponse,
    ReplyRequest, ReplyResponse,
)
from brain.generator.reply import ReplyGenerator
from brain.media.emoji_suggester import suggest_reactions
from brain.media.gif_suggester import build_gif_query, search_gifs
from brain.memory.store import PersonaStore
from brain.profiler.analyzer import load_profile, profile_path

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

CHROMA_PATH  = os.getenv("CHROMA_PATH",  "data/db/chroma")
PROFILES_DIR = os.getenv("PROFILES_DIR", "data/profiles")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
API_HOST     = os.getenv("API_HOST",     "127.0.0.1")
API_PORT     = int(os.getenv("API_PORT", "8765"))

# ── Global singletons ──────────────────────────────────────────────────────────
# Heavy: embedding model load + ChromaDB init. Loaded once at startup.

_store:     PersonaStore | None  = None
_generator: ReplyGenerator | None = None
_executor   = ThreadPoolExecutor(max_workers=4)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load singletons at startup. Shut down thread pool at shutdown."""
    global _store, _generator

    print(f"  [PersonaOS] Loading store from {CHROMA_PATH}...")
    _store = PersonaStore(chroma_path=CHROMA_PATH)
    print(f"  [PersonaOS] Store ready — {_store.count()} messages indexed")

    print(f"  [PersonaOS] Loading ReplyGenerator ({OLLAMA_MODEL})...")
    _generator = ReplyGenerator(_store, profiles_dir=PROFILES_DIR, model=OLLAMA_MODEL)
    print(f"  [PersonaOS] Brain API ready → http://{API_HOST}:{API_PORT}")

    yield

    _executor.shutdown(wait=False)
    print("  [PersonaOS] Shutdown complete")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PersonaOS Brain API",
    description="Local-first communication intelligence. All processing stays on your device.",
    version="0.3.5",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # local only — no auth required
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_profile_safe(user_name: str, contact: str = ""):
    """
    Load profile + relationship for a user.
    Returns (profile, relationship) or (None, None) on any failure.
    Never raises — callers degrade gracefully without a profile.
    """
    if not user_name:
        return None, None
    try:
        p = load_profile(profile_path(user_name, PROFILES_DIR))
        rel = None
        if contact and p.relationships:
            cl = contact.lower()
            for r in p.relationships:
                if r.contact_name.lower() == cl or cl in r.contact_name.lower():
                    rel = r
                    break
        return p, rel
    except (FileNotFoundError, ValueError):
        return None, None


async def _in_executor(fn):
    """Run a blocking function in the thread pool without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Liveness check. < 1ms."""
    return HealthResponse(
        status="ok",
        store_size=_store.count() if _store else 0,
        profile_dir=PROFILES_DIR,
        ollama_model=OLLAMA_MODEL,
    )


@app.post("/suggest/reply", response_model=ReplyResponse, tags=["suggest"])
async def suggest_reply(req: ReplyRequest):
    """
    Generate personalized reply drafts for an incoming message.
    Calls Ollama locally — expect 5-10s response time.
    """
    if not _generator:
        raise HTTPException(503, "Generator not initialised — server still starting up")

    try:
        drafts, explainability = await _in_executor(
            lambda: _generator.generate(
                incoming=req.incoming,
                from_contact=req.from_contact,
                user_name=req.user_name,
                num_drafts=req.num_drafts,
            )
        )
    except RuntimeError as e:
        raise HTTPException(503, f"Ollama error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Reply generation failed: {e}")

    # drafts == [] is a valid 200 — smart silence. The reason rides in
    # explainability["silence_skip"] / ["silence_reason"], same as the CLI.
    return ReplyResponse(
        drafts=list(drafts),
        model=req.model,
        store_size=_store.count() if _store else 0,
        contact=req.from_contact,
        explainability=explainability,
    )


@app.post("/suggest/reaction", response_model=ReactionResponse, tags=["suggest"])
async def suggest_reaction(req: ReactionRequest):
    """
    Suggest emoji reactions for an incoming message.
    Pure local heuristic — < 20ms, zero external calls.
    """
    profile, relationship = _load_profile_safe(req.user_name, req.contact)

    # Synchronous: fast enough to run directly on the event loop
    reactions = suggest_reactions(
        req.incoming,
        profile=profile,
        relationship=relationship,
        top_k=req.top_k,
    )

    return ReactionResponse(
        reactions=[
            ReactionItem(
                emoji=r.emoji,
                confidence=r.confidence,
                reason=r.reason,
                source=r.source,
            )
            for r in reactions
        ],
        profile_loaded=profile is not None,
    )


@app.post("/suggest/gif", response_model=GifResponse, tags=["suggest"])
async def suggest_gif(req: GifRequest):
    """
    Search Giphy for relevant GIFs.
    Requires GIPHY_API_KEY in .env. Returns empty list if key absent.
    100-500ms when key present.
    """
    _, relationship = _load_profile_safe(req.user_name, req.contact)
    query = build_gif_query(req.incoming, relationship)

    gifs = await _in_executor(
        lambda: search_gifs(req.incoming, relationship=relationship, limit=req.limit)
    )

    return GifResponse(
        gifs=[
            GifItem(
                title=g.title,
                url=g.url,
                preview_url=g.preview_url,
                source=g.source,
            )
            for g in gifs
        ],
        query_used=query,
        count=len(gifs),
    )


@app.get("/profile/{name}", response_model=ProfileResponse, tags=["profile"])
async def get_profile(name: str):
    """
    Load a saved communication profile by name.
    Pure JSON read — < 5ms.
    """
    try:
        profile = load_profile(profile_path(name, PROFILES_DIR))
    except FileNotFoundError:
        raise HTTPException(404, f"No profile found for '{name}'. Run: python -m cli.main setup")
    except ValueError as e:
        raise HTTPException(422, str(e))

    style = profile.global_style
    emoji = profile.global_emoji

    return ProfileResponse(
        name=name,
        total_messages=profile.total_messages_analyzed,
        platform=profile.platform,
        relationships=[
            ProfileRelationship(
                contact_name=r.contact_name,
                relationship_type=r.relationship_type,
                confidence=r.relationship_confidence,
                message_count=r.message_count,
                top_signal=r.relationship_signals[0] if r.relationship_signals else "",
            )
            for r in profile.relationships[:20]
        ],
        avg_message_length=style.avg_message_length if style else 0.0,
        top_emoji=[e for e, _ in (emoji.top_emojis[:5] if emoji else [])],
        slang_tokens=style.slang_tokens[:10] if style else [],
    )
