"""
tests/test_api.py

Tests for Phase 3.5 — Brain API Server.

Strategy:
  - /health and /suggest/reaction run without mocking (pure local)
  - /suggest/gif tested without API key (returns empty, not error)
  - /suggest/reply mocks ReplyGenerator to avoid Ollama dependency
  - /profile/{name} uses tmp_path with a real serialized profile

All tests are fast (no Ollama, no network).
Integration tests that need Ollama are marked @pytest.mark.integration.

Run:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v -m "not integration"
"""
import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from brain.profiler.models import (
    CommunicationProfile, StyleProfile, EmojiProfile, RelationshipProfile,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_minimal_profile(tmp_path: Path, name: str = "Rishit Babbar") -> Path:
    """Write a minimal serialised CommunicationProfile to tmp_path."""
    profile = CommunicationProfile(
        platform="whatsapp",
        total_messages_analyzed=500,
        global_style=StyleProfile(
            avg_message_length=45.0,
            avg_word_count=8.0,
            uses_lowercase_heavily=True,
            slang_tokens=["lol", "fr", "ngl"],
        ),
        global_emoji=EmojiProfile(
            total_emoji_count=80,
            top_emojis=[("😂", 40), ("🔥", 20)],
            emoji_frequency=0.35,
        ),
        relationships=[
            RelationshipProfile(
                contact_name="Priya",
                message_count=300,
                relationship_type="close_friend",
                relationship_confidence=0.90,
                relationship_signals=["plans meetups regularly"],
            )
        ],
    )
    (tmp_path / "profiles").mkdir(exist_ok=True)
    p = tmp_path / "profiles" / "rishit_babbar.json"
    p.write_text(profile.model_dump_json(), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    TestClient with heavy singletons patched out.
    PersonaStore and ReplyGenerator constructors are replaced with mocks
    so the lifespan completes without touching disk or Ollama.

    Order matters: importlib.reload() re-executes the module and rebinds
    ReplyGenerator/PersonaStore from their real imports, which silently undoes
    any patch applied before it. So env vars -> reload -> patch names ->
    start client (lifespan runs inside the patches).
    """
    tmp = tmp_path_factory.mktemp("api_test")
    _make_minimal_profile(tmp)

    mock_store = MagicMock()
    mock_store.count.return_value = 100

    mock_gen = MagicMock()
    # Mirrors the real v6 contract: ReplyGenerator.generate() -> (drafts, explainability).
    # Keys match ExplainabilityRecord.to_dict() so drift there surfaces here.
    mock_gen.generate.return_value = (
        [
            "haan yaar chalte hain",
            "theek hai bro sounds good",
            "lol yes fr im free",
        ],
        {
            "intent": "question",
            "emotion": "neutral",
            "stage": "unknown",
            "language": "hinglish",
            "relationship_type": "close_friend",
            "relationship_label": "close friend",
            "retrieved_examples_count": 2,
            "retrieved_example_previews": ["chalte hain aaj?", "haan bhai pkka"],
            "has_profile": True,
            "has_relationship": True,
            "silence_skip": False,
            "silence_reason": "",
        },
    )

    # PROFILES_DIR is read at import time, so env must be set before the reload.
    with patch.dict("os.environ", {
        "CHROMA_PATH":  str(tmp / "chroma"),
        "PROFILES_DIR": str(tmp / "profiles"),
    }):
        import importlib
        import api.server as srv
        importlib.reload(srv)

        with patch("api.server.PersonaStore", return_value=mock_store), \
             patch("api.server.ReplyGenerator", return_value=mock_gen):
            assert srv.ReplyGenerator.return_value is mock_gen, "patch did not survive reload"
            assert srv.PersonaStore.return_value is mock_store, "patch did not survive reload"
            with TestClient(srv.app) as c:
                yield c, tmp


@pytest.fixture
def gen_mock(client):
    """
    The mocked ReplyGenerator instance behind /suggest/reply.

    The client fixture is module-scoped, so tests that need a different
    generate() return value must go through this fixture: it restores the
    original return_value on teardown, keeping the shared mock clean
    regardless of test ordering.
    """
    import api.server as srv
    gen = srv._generator
    assert gen is not None, "lifespan must have installed the mocked generator"
    original = gen.generate.return_value
    yield gen
    gen.generate.return_value = original


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_returns_200(self, client):
        c, _ = client
        r = c.get("/health")
        assert r.status_code == 200

    def test_status_is_ok(self, client):
        c, _ = client
        assert r.json()["status"] == "ok" if (r := c.get("/health")) else False

    def test_response_has_required_fields(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert "status" in data
        assert "store_size" in data
        assert "ollama_model" in data

    def test_store_size_is_integer(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert isinstance(data["store_size"], int)


# ── /suggest/reaction ──────────────────────────────────────────────────────────

class TestReactionEndpoint:
    """Pure local heuristic — no mocking needed."""

    def test_returns_200(self, client):
        c, _ = client
        r = c.post("/suggest/reaction", json={"incoming": "lmao bhai 😂"})
        assert r.status_code == 200

    def test_returns_reactions_list(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": "lmao bro that was hilarious"}).json()
        assert "reactions" in data
        assert isinstance(data["reactions"], list)

    def test_humor_message_gets_humor_emojis(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": "lmao fr dead 💀"}).json()
        emojis = [r["emoji"] for r in data["reactions"]]
        humor_set = {"😂", "💀", "🤣", "😭", "😅"}
        assert any(e in humor_set for e in emojis)

    def test_celebration_message_gets_celebration_emojis(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": "i got the internship!!"}).json()
        emojis = [r["emoji"] for r in data["reactions"]]
        assert any(e in {"🎉", "🥳", "🔥", "💯", "🙌"} for e in emojis)

    def test_empty_message_returns_empty_reactions(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": ""}).json()
        assert data["reactions"] == []

    def test_top_k_respected(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": "hi there", "top_k": 3}).json()
        assert len(data["reactions"]) <= 3

    def test_confidence_values_in_range(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": "lmao bro"}).json()
        for r in data["reactions"]:
            assert 0.0 <= r["confidence"] <= 1.0

    def test_profile_loaded_false_without_user(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": "hey"}).json()
        assert data["profile_loaded"] is False

    def test_each_reaction_has_required_fields(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": "lmao"}).json()
        for r in data["reactions"]:
            assert "emoji" in r
            assert "confidence" in r
            assert "reason" in r
            assert "source" in r

    def test_no_duplicate_emojis_in_response(self, client):
        c, _ = client
        data = c.post("/suggest/reaction", json={"incoming": "funny lol", "top_k": 20}).json()
        emojis = [r["emoji"] for r in data["reactions"]]
        assert len(emojis) == len(set(emojis))

    def test_incoming_too_long_rejected(self, client):
        c, _ = client
        r = c.post("/suggest/reaction", json={"incoming": "x" * 3000})
        assert r.status_code == 422


# ── /suggest/gif ───────────────────────────────────────────────────────────────

class TestGifEndpoint:

    def test_returns_200_without_api_key(self, client, monkeypatch):
        c, _ = client
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        r = c.post("/suggest/gif", json={"incoming": "that was hilarious"})
        assert r.status_code == 200

    def test_empty_gifs_without_api_key(self, client, monkeypatch):
        c, _ = client
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        data = c.post("/suggest/gif", json={"incoming": "lmao"}).json()
        assert data["gifs"] == []
        assert data["count"] == 0

    def test_query_used_is_present(self, client, monkeypatch):
        c, _ = client
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        data = c.post("/suggest/gif", json={"incoming": "that was hilarious bro"}).json()
        assert isinstance(data["query_used"], str)
        assert len(data["query_used"]) > 0

    def test_limit_field_accepted(self, client, monkeypatch):
        c, _ = client
        monkeypatch.delenv("GIPHY_API_KEY", raising=False)
        r = c.post("/suggest/gif", json={"incoming": "hi", "limit": 3})
        assert r.status_code == 200

    def test_limit_too_large_rejected(self, client):
        c, _ = client
        r = c.post("/suggest/gif", json={"incoming": "hi", "limit": 100})
        assert r.status_code == 422


# ── /suggest/reply ─────────────────────────────────────────────────────────────

class TestReplyEndpoint:

    def test_returns_200(self, client):
        c, _ = client
        r = c.post("/suggest/reply", json={
            "incoming": "are you free tonight?",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
        })
        assert r.status_code == 200

    def test_returns_drafts_list(self, client):
        c, _ = client
        data = c.post("/suggest/reply", json={
            "incoming": "bhai kya scene hai",
            "from_contact": "Shreyansh",
            "user_name": "Rishit Babbar",
        }).json()
        assert "drafts" in data
        assert isinstance(data["drafts"], list)
        assert len(data["drafts"]) > 0

    def test_drafts_are_non_empty_strings(self, client):
        c, _ = client
        data = c.post("/suggest/reply", json={
            "incoming": "hi",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
        }).json()
        for d in data["drafts"]:
            assert isinstance(d, str) and len(d) > 0

    def test_contact_echoed_in_response(self, client):
        c, _ = client
        data = c.post("/suggest/reply", json={
            "incoming": "hi",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
        }).json()
        assert data["contact"] == "Priya"

    def test_store_size_in_response(self, client):
        c, _ = client
        data = c.post("/suggest/reply", json={
            "incoming": "hi",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
        }).json()
        assert isinstance(data["store_size"], int)

    def test_missing_required_fields_returns_422(self, client):
        c, _ = client
        r = c.post("/suggest/reply", json={"incoming": "hi"})
        assert r.status_code == 422

    def test_num_drafts_field_accepted(self, client):
        c, _ = client
        r = c.post("/suggest/reply", json={
            "incoming": "hi",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
            "num_drafts": 5,
        })
        assert r.status_code == 200

    def test_num_drafts_too_large_rejected(self, client):
        c, _ = client
        r = c.post("/suggest/reply", json={
            "incoming": "hi",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
            "num_drafts": 50,
        })
        assert r.status_code == 422

    def test_generator_tuple_is_unpacked_into_drafts(self, client):
        """generate() returns (drafts, explainability) since prompt v6."""
        c, _ = client
        data = c.post("/suggest/reply", json={
            "incoming": "are you free tonight?",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
        }).json()
        assert data["drafts"] == [
            "haan yaar chalte hain",
            "theek hai bro sounds good",
            "lol yes fr im free",
        ]

    def test_explainability_key_always_present(self, client):
        c, _ = client
        data = c.post("/suggest/reply", json={
            "incoming": "hi",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
        }).json()
        assert "explainability" in data

    def test_explainability_populated(self, client):
        c, _ = client
        data = c.post("/suggest/reply", json={
            "incoming": "bhai free hai kal?",
            "from_contact": "Shreyansh",
            "user_name": "Rishit Babbar",
        }).json()
        expl = data["explainability"]
        assert isinstance(expl, dict)
        assert expl["language"] == "hinglish"
        assert expl["intent"] == "question"
        assert expl["has_profile"] is True
        assert expl["silence_skip"] is False

    def test_explainability_passes_through_unmodified(self, client, gen_mock):
        """The whole record reaches the client — no key dropped or rewritten."""
        c, _ = client
        expected = {"lang": "hinglish"}
        gen_mock.generate.return_value = (["a", "b"], expected)
        data = c.post("/suggest/reply", json={
            "incoming": "bhai kya scene hai",
            "from_contact": "Shreyansh",
            "user_name": "Rishit Babbar",
        }).json()
        assert data["drafts"] == ["a", "b"]
        assert data["explainability"] == expected

    def test_smart_silence_returns_200_with_empty_drafts(self, client, gen_mock):
        """
        Smart silence is a valid 200, not an error. The client distinguishes it
        from a broken generator via explainability["silence_skip"].
        """
        c, _ = client
        gen_mock.generate.return_value = ([], {
            "intent": "general",
            "emotion": "neutral",
            "stage": "unknown",
            "language": "hinglish",
            "relationship_type": "unknown",
            "relationship_label": "friend",
            "retrieved_examples_count": 0,
            "retrieved_example_previews": [],
            "has_profile": False,
            "has_relationship": False,
            "silence_skip": True,
            "silence_reason": "low-value acknowledgment message",
        })
        r = c.post("/suggest/reply", json={
            "incoming": "ok",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["drafts"] == []
        assert data["explainability"]["silence_skip"] is True
        assert data["explainability"]["silence_reason"] == "low-value acknowledgment message"

    def test_explainability_null_when_generator_omits_it(self, client, gen_mock):
        c, _ = client
        gen_mock.generate.return_value = (["only draft"], None)
        r = c.post("/suggest/reply", json={
            "incoming": "are you free tonight?",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["drafts"] == ["only draft"]
        assert data["explainability"] is None

    def test_existing_fields_unchanged_by_explainability(self, client, gen_mock):
        """Backward compatibility: model / store_size / contact keep their shape."""
        c, _ = client
        gen_mock.generate.return_value = (["a"], {"intent": "question"})
        data = c.post("/suggest/reply", json={
            "incoming": "hi",
            "from_contact": "Priya",
            "user_name": "Rishit Babbar",
            "model": "qwen3.5:9b",
        }).json()
        assert data["model"] == "qwen3.5:9b"
        assert data["contact"] == "Priya"
        assert isinstance(data["store_size"], int)


# ── /profile/{name} ────────────────────────────────────────────────────────────

class TestProfileEndpoint:

    def test_returns_200_for_existing_profile(self, client):
        c, _ = client
        r = c.get("/profile/Rishit Babbar")
        assert r.status_code == 200

    def test_returns_404_for_missing_profile(self, client):
        c, _ = client
        r = c.get("/profile/CompletelyUnknownPerson99999")
        assert r.status_code == 404

    def test_response_has_required_fields(self, client):
        c, _ = client
        data = c.get("/profile/Rishit Babbar").json()
        assert "name" in data
        assert "total_messages" in data
        assert "platform" in data
        assert "relationships" in data

    def test_relationships_is_list(self, client):
        c, _ = client
        data = c.get("/profile/Rishit Babbar").json()
        assert isinstance(data["relationships"], list)

    def test_relationship_has_required_fields(self, client):
        c, _ = client
        data = c.get("/profile/Rishit Babbar").json()
        if data["relationships"]:
            rel = data["relationships"][0]
            assert "contact_name" in rel
            assert "relationship_type" in rel
            assert "confidence" in rel
            assert "message_count" in rel

    def test_avg_message_length_is_float(self, client):
        c, _ = client
        data = c.get("/profile/Rishit Babbar").json()
        assert isinstance(data["avg_message_length"], float)

    def test_top_emoji_is_list(self, client):
        c, _ = client
        data = c.get("/profile/Rishit Babbar").json()
        assert isinstance(data["top_emoji"], list)


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests — require Ollama running
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestReplyEndpointIntegration:
    """
    Run: pytest tests/test_api.py -m integration -v -s
    Requires: ollama serve + qwen3.5:9b pulled
    """

    def test_generates_real_drafts(self):
        """End-to-end: real store, real Ollama, no mocks."""
        from api.server import app
        with TestClient(app) as c:
            r = c.post("/suggest/reply", json={
                "incoming": "bhai free hai kal?",
                "from_contact": "Shreyansh",
                "user_name": "Rishit Babbar",
            })
            assert r.status_code == 200
            data = r.json()
            assert len(data["drafts"]) > 0
            for d in data["drafts"]:
                assert isinstance(d, str) and len(d) > 0
