# PersonaOS

> A privacy-first Communication Intelligence Engine that learns how you communicate.

**Not an AI chatbot. Not an auto-reply bot. A communication copilot.**

PersonaOS learns how you write — your slang, emoji habits, message length, tone with each contact, Hinglish patterns, relationship dynamics — and uses that knowledge to help you draft replies that sound like *you*.

---

## What it does

```
Your chat exports
      ↓
Parse → Profile → Classify relationships
      ↓
Embed messages into local vector store
      ↓
Incoming message → Retrieve similar past messages → Draft reply via Ollama
      ↓
You review and approve. Nothing sends automatically.
```

All of this runs **locally on your machine**. No cloud. No API keys. No data leaving your device.

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Communication Profiler — parse exports → style profile → relationship classification | ✅ Complete |
| **2A** | Memory Store — embed messages into ChromaDB for semantic retrieval | ✅ Complete |
| **2B** | Reply Generator — RAG + Ollama → personalized draft replies | ✅ Complete |
| **3A** | Media Intelligence — GIF / Reaction/emoji/reaction suggestions| 🔨 Building |
| **3B** | Media Intelligence — Memes| Planned |
| **3C** | Media Intelligence — Stickers| Planned |
| 4 | Android Keyboard integration | Planned |
| 5 | On-device inference | Planned |

[Notion](https://app.notion.com/p/PersonaOS-376e9284616580869547f38f4a88fd46?source=copy_link)
[Miro](https://miro.com/app/board/uXjVHJ4WsaU=/)

---

## Architecture

```
brain/
├── profiler/
│   ├── models.py          Pydantic data contracts (MessageRecord, CommunicationProfile)
│   ├── parsers/
│   │   ├── whatsapp.py    WhatsApp .txt export parser (Indian/Android format supported)
│   │   └── telegram.py    Telegram JSON export parser
│   ├── analyzer.py        Style extraction → CommunicationProfile
│   ├── signals.py         Conversation signal extraction (phrases, behaviors, topics, emoji)
│   └── classifier.py      Relationship scoring from ConversationSignals
├── memory/
│   └── store.py           ChromaDB vector store with multilingual embeddings
└── generator/
    └── reply.py           Reply generation (Phase 2B — in progress)
```

### How relationship classification works

The classifier does not count keywords. It extracts **structured signals** from the full conversation:

- **Phrase patterns** — "bhai assignment bhej de" → `college_friend` (not split into family + academic)
- **Behavioral patterns** — consistent check-ins, planning meetups, emotional support
- **Topic clusters** — what the conversation is *about* (academic / social / logistics / emotional)
- **Session dynamics** — who initiates, how deep sessions get, contact frequency
- **Emoji semantics** — humor emojis (😂💀) vs affection emojis (❤️🥰) scored differently
- **Hinglish normalisation** — `nhi` → `nahi`, `yr` → `yaar`, `kr` → `kar` (40+ mappings)

Relationship types: `close_friend` · `college_friend` · `family` · `romantic_interest` · `colleague` · `acquaintance`

---

## Setup

**Requirements:** Python 3.12, [Ollama](https://ollama.ai) running locally with `qwen3.5:9b`

```bash
git clone https://github.com/your-username/PersonaOS.git
cd PersonaOS

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python scripts/setup_check.py
```

Pull the local model (one-time, ~6GB):

```bash
ollama pull qwen3.5:9b
```

---

## Usage

### Phase 1 — Build your communication profile

Export a WhatsApp chat: **Chat → ⋮ → More → Export Chat → Without Media**

```bash
# Parse exports and build profile (pass a directory or individual files)
python -m cli.main setup data/exports/ --you "Your Name"

# View your profile with relationship breakdown
python -m cli.main profile "Your Name"

# Deep-dive signal inspection for a specific contact
python -m cli.main signals "data/exports/Chat with Alex.txt" --you "Your Name"

# Generate natural language summary (requires: ollama serve)
python -m cli.main summary "Your Name"
```

### Phase 2A — Populate semantic memory

```bash
# Embed your messages into ChromaDB
python -m cli.main store data/exports/ --you "Your Name"

# Test semantic retrieval
python -m cli.main search "are you free tonight?" --you "Your Name" --top 5
```

### Diagnostics

```bash
python -m cli.main check          # environment check
python scripts/diagnose.py        # full diagnostic
python scripts/project_audit.py   # generate project state snapshot
```

---

## Project Structure

```
PersonaOS/
├── brain/                  Core intelligence modules
│   ├── profiler/           Phase 1 — parse and profile
│   ├── memory/             Phase 2A — vector memory store
│   └── generator/          Phase 2B — reply generation (in progress)
├── cli/
│   └── main.py             8-command CLI (Typer + Rich)
├── data/
│   ├── exports/            Your chat files (gitignored)
│   ├── profiles/           Generated JSON profiles (gitignored)
│   └── db/                 ChromaDB + SQLite (gitignored)
├── tests/                  57+ tests (pytest)
│   ├── test_parsers.py
│   ├── test_store.py
│   └── test_signals.py
└── scripts/
    ├── setup_check.py
    ├── diagnose.py
    └── project_audit.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Data models | Pydantic v2 |
| CLI | Typer + Rich |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) |
| Vector store | ChromaDB (local SQLite) |
| Local LLM | Ollama — `qwen3.5:9b` |
| Testing | pytest |
| Platform | macOS Apple Silicon (arm64) |

---

## Privacy

- **All processing is local.** No data ever leaves your machine.
- **Chat exports are gitignored.** They are never committed, not even accidentally.
- **No cloud LLM calls.** Ollama runs the model on your hardware.
- **No accounts, no telemetry, no API keys required.**

---

## Running Tests

```bash
pytest tests/ -v
# Expected: 57+ tests passing across parsers, store, signals, and classifier
```

---

## Why not just use ChatGPT?

Generic AI writes generically. PersonaOS is trained on *your* communication — your slang, your Hinglish patterns, your emoji register, your relationship dynamics. A reply to your college friend sounds different from a reply to your manager. PersonaOS knows the difference because it has seen thousands of your actual messages.

And it does all of this without sending a single message to any server.
