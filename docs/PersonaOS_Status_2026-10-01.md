# PersonaOS — Status Report & Re-entry Plan
*Generated 2026-10-01 from the uploaded `PersonaOS.zip` (source, git history, tests only — I did not read anything under `data/`).*

---

## 1. Where you actually are

Last commit: **2026-07-26** (`5a5a6e1`). Working tree is **clean**, branch `master`, in sync with `origin/master`. You stopped right after a stabilization sprint — and it mostly worked.

### Sprint 3.5.1 bug queue — what the git log says

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | `fastapi` missing from venv | ✅ Fixed | `requirements.txt` now has `fastapi>=0.110,<=0.115`, `uvicorn`, `httpx<0.28`; `test_api.py` collects and runs |
| 2 | ChromaDB silent empty results | ✅ Fixed (code) | `83444c3`: chat_name matching (canonical + legacy prefixed), time-range filtering moved to Python, prefix stripped at write time |
| 3 | WhatsApp filename prefix regex | ✅ Fixed | `83444c3`: case-insensitive strip, applied to filename-derived and explicit names |
| 4 | `diagnose_reply.py` diverged | ✅ Fixed | `83444c3`: now imports and calls the real `build_prompt()` |
| 5 | "Dev" → family misclassification | ✅ Fixed | `9a3bf01`: honorific terms (bhaiya) no longer override professional-topic signals; +273 lines of tests |
| 6 | Git hygiene | ✅ Fixed | `.gitignore` hardened, duplicate `custom/profiles/rishit_babbar.json` deleted, no `data/` or `custom/` files tracked |

**Caveat on #2:** the fix is verified by unit tests, but nobody has re-run your original repro (`python -m cli.main search "plans for the weekend" --you "Rishit Babbar" --chat "Vibhu Kakkar"`) against your real 4,130-message store. A silent-failure bug isn't closed until the real query returns results. Existing stores written under the old naming may also need a re-ingest.

### Test run (my sandbox, sentence-transformers stubbed, no Ollama)

**392 passed, 12 failed, 4 integration deselected.** I could not run `test_store.py` (needs the real embedding model download, blocked here). All 12 failures are real and explained below.

---

## 2. What's still broken (new findings — not in your old bug list)

### NEW-1 (HIGH): `/suggest/reply` is broken against the real generator
`ReplyGenerator.generate()` returns a **tuple** `(drafts, explainability)` since prompt v6 (`intent_sensors.py` work). `api/server.py:158` still does `drafts = generate(...)` and passes the whole tuple into `ReplyResponse(drafts: list[str])`. Against the real generator this will 500. This is the Phase 3.5 endpoint Phase 4 depends on.
- `generate_legacy()` exists for exactly this and returns `list[str]`; either use it, or unpack the tuple and also expose the explainability/smart-silence info in the response.
- Design decision for you: Phase 4 keyboards benefit from `explainability` (language, intent, "smart silence" = don't suggest). I'd unpack and surface it rather than hide it behind legacy.

### NEW-2 (HIGH): the API test fixture doesn't actually mock the generator
`tests/test_api.py` patches `api.server.ReplyGenerator`, then calls `importlib.reload(srv)`. Reload re-runs `from brain.generator.reply import ReplyGenerator`, **overwriting the patch with the real class**. So 6 `TestReplyEndpoint` tests hit real Ollama (fail with 503 when it's down), and the mock's list-return shape never matched the v6 tuple anyway. Fix: drop the reload, or patch `brain.generator.reply.ReplyGenerator` instead, and make the mock return `(drafts, {})`.
- Your `.pytest_cache/lastfailed` shows these same 6 + the integration test failing on your Mac too — so this predates the break in my sandbox.

### NEW-3 (MED): "NoOllama" smoke tests need Ollama
`TestReplyGeneratorV6NoOllama` (6 tests) uses an empty store, but `generate()` still calls the LLM, so they raise `RuntimeError: Ollama call failed` when Ollama is down. They're mislabeled as no-Ollama and not marked `@pytest.mark.integration`. Either mock `ollama.generate` or mark them integration. This is the "tests that pass vacuously / lie about their dependencies" class of problem from your own learnings.

### NEW-4 (LOW): stale tracking files
- `bugs.txt` / `bugs.md` still list the original 3 bugs, all now fixed. Archive or clear them.
- `bugs.md` is gitignored but `bugs.txt` is not — and it contains real contact names and a chat search transcript. **Check `git ls-files bugs.txt`**; if tracked, it's in your public repo history. (`git status` is clean, so it's either tracked or covered; verify.)
- `readme.md` / Miro / Notion: still the Phase 3.5.0 doc-sync debt.
- `.gitignore` ignores `opencode.json` and `PersonaOS_ProjectPlan.md`, which is fine, but note the zip also contains `.env`, `.claude-flow/`, `.sixth/` — never share the zip beyond this chat.

### Privacy flag
The zip you uploaded contained `data/exports/` (12 real WhatsApp chats), `data/db/chroma/`, and `data/profiles/`. I didn't open them. For future uploads, exclude `data/` (`zip -r PersonaOS.zip PersonaOS -x "PersonaOS/data/*" "PersonaOS/.env" "PersonaOS/.venv/*"`). This is the same guardrail you apply to OpenCode.

---

## 3. What's remaining, in priority order

1. **Verify bug #2 on real data** (5 min, your Mac): rerun the `search` repro; if empty, re-ingest.
2. **NEW-1** fix server tuple handling (+ decide whether to expose explainability).
3. **NEW-2 / NEW-3** repair the tests so there's a *true* green baseline with Ollama off (`pytest -m "not integration"`) and a separate green run with Ollama on.
4. **Run `test_store.py` on your Mac** — I couldn't; commit message claims 63/63 for the targeted tests only.
5. **Phase 3.5.0 doc sync** — README, Miro "Current Status", GIF flow doc (Tenor → Giphy), Notion.
6. **Real-world API smoke test**: `python -m cli.main serve`, then `curl` all 5 endpoints with Ollama up. A green mocked suite isn't proof the server works (NEW-1 is the proof).
7. **Gate to Phase 4:** only after 1–6. Then spec Phase 4 v1 (single app, manual contact picker, LAN-only, no auto-send) and **measure phone → LAN → Mac Ollama latency before designing any UX**.
8. **Phase 2C (Recap/Note Rewriter):** still needs your explicit scoping decision — add or park. My recommendation: park until Phase 4 v1 is specced; it doesn't unblock anything.
9. Parked, unchanged: 3B memes (needs taxonomy), 3C stickers (needs platform choice), Phase 5 on-device (separate scoping).

---

## 4. OpenCode task specs

### Task A — Fix `/suggest/reply` tuple contract
```
MODEL: nvidia nim qwen/qwen3-coder-480b-a35b-instruct
CONTEXT: ReplyGenerator.generate() returns tuple[list[str], dict] since prompt v6.
         api/server.py suggest_reply (≈line 158) treats the return as list[str].
FILES: api/server.py, api/models.py
OBJECTIVE: Unpack (drafts, explain). Return drafts as list[str]. Add optional
           `explainability: dict | None` to ReplyResponse and populate it.
           Handle empty drafts (smart-silence) as a valid 200 response.
CONSTRAINTS: Source code only. Do not read data/, custom/. Keep existing fields
             backward compatible. No new dependencies.
ACCEPTANCE: Calling the endpoint with a generator returning (["a","b"], {"lang":"hinglish"})
            yields 200 with drafts==["a","b"] and explainability populated.
```

### Task B — Repair API + smoke tests
```
MODEL: OpenCode Zen — DeepSeek V4 Flash Free
CONTEXT: tests/test_api.py client fixture patches api.server.ReplyGenerator then
         importlib.reload(srv), which discards the patch. Mock returns list, not tuple.
         tests/test_intent_sensors.py::TestReplyGeneratorV6NoOllama calls the real LLM.
FILES: tests/test_api.py, tests/test_intent_sensors.py
OBJECTIVE: (1) Remove reload or patch brain.generator.reply.ReplyGenerator; mock returns
           (drafts, {}). (2) In the NoOllama class, monkeypatch ollama.generate to return
           a canned JSON reply, or mark tests @pytest.mark.integration.
CONSTRAINTS: Keep len(drafts) > 0 assertions explicit. No vacuous passes.
ACCEPTANCE: `pytest -m "not integration"` passes with Ollama stopped.
```

---

## 5. Commit message (for the next session's work)
```
fix(api): unpack v6 generate() tuple in /suggest/reply; repair API/smoke tests

- server: handle (drafts, explainability), expose explainability, allow empty drafts
- tests: remove fixture reload that discarded ReplyGenerator patch; mock v6 tuple
- tests: mock Ollama in V6NoOllama smoke tests
- chore: clear resolved entries from bugs.txt
```

---

## 6. Notion update (copy-paste)
```
[2026-10-01] Session: Re-entry audit after ~10-week gap

- State: working tree clean on master @ 5a5a6e1 (2026-07-26). Sprint 3.5.1
  bugs 1–6 resolved in code (fastapi deps, ChromaDB chat_name matching,
  WhatsApp prefix strip, diagnose_reply uses real build_prompt, bhaiya
  classifier fix, git hygiene/duplicate profile removed).
- Issue found: /suggest/reply broken vs real generator — v6 generate() returns
  (drafts, explain) tuple, server expects list. Hidden because the API test
  fixture's importlib.reload() discards the ReplyGenerator mock.
  Also: 6 "NoOllama" smoke tests actually call Ollama.
- Test result in sandbox: 392 pass / 12 fail (all explained above); test_store
  not run (needs embedding model).
- Lesson: a green mocked suite != working server. Reload-after-patch silently
  voids mocks; test names must match real dependencies.
- Still open: verify search repro on real store, doc sync (README/Miro/Notion),
  Phase 2C scoping decision, then Phase 4 v1 spec (latency measured first).
- Privacy: uploaded zip contained data/ — exclude it in future uploads.
- Next: Task A (server tuple fix) -> Task B (tests) -> real `serve` + curl smoke
  test -> doc sync -> Phase 4 v1 spec.
```

## 7. Miro update
- **Current Status frame:** Phase 3.5 = "built, 2 known defects (reply tuple contract, test fixture)". Phase 3.5.1 bugs → mark 6/6 fixed in code.
- **Phase 3A / GIF flow:** replace Tenor with Giphy.
- **Add card:** "Verify search fix on real store".
- **Phase 2C card:** "Pending scoping — recommend park".
(Small batches, ~15–20 items per call, draw within-frame connectors manually.)
