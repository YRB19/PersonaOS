# PersonaOS

> A privacy-first personal communication brain that learns how you communicate.

[Notion](https://app.notion.com/p/PersonaOS-376e9284616580869547f38f4a88fd46?source=copy_link)
[Visit OpenAI](https://openai.com)

**Status: Phase 1 — in development**

---

## What it does

PersonaOS learns how you write — your slang, emoji habits, message length, tone per contact — and uses that knowledge to help you communicate faster and more authentically.

It is **not** an auto-reply bot. It is a communication copilot.

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_check.py
```

Requires Ollama running locally with `qwen3.5:9b`.

---

## Usage (Phase 1)

```bash
# Parse a WhatsApp export
python cli/main.py parse data/exports/your_chat.txt --you "Your Name"

# Run environment check
python cli/main.py check

# Generate project state snapshot (for AI context)
python scripts/project_audit.py
```

---

## Project state

Run `python scripts/project_audit.py` to generate `project_state.txt`.  
Paste this file into your AI assistant for full codebase context.

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Communication Profile Generator (parse exports → style profile) | 🔨 In progress |
| 2 | Communication Brain (RAG memory + reply generation) | Planned |
| 3 | Media Intelligence (GIF/meme/sticker suggestions) | Planned |
| 4 | Android Keyboard | Planned |
| 5 | On-device inference | Planned |

---

## Privacy

- All processing is local. No data leaves your machine.
- Chat exports are gitignored. Never committed.
- Local Ollama models. No cloud LLM calls in Phase 1.

---

## Stack

- **Python 3.12** + venv
- **Ollama** (`qwen3.5:9b`) — local LLM
- **ChromaDB** — local vector store (Phase 2)
- **Sentence Transformers** — local embeddings (Phase 2)
- **Pydantic v2** — data models
- **Rich + Typer** — CLI
