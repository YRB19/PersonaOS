# PersonaOS Command Reference

Complete reference for all PersonaOS CLI commands and utility scripts.

---

## CLI Commands

All commands are invoked as:

```bash
python -m cli.main <command> [options]
```

---

## setup

**Description:** Parse one or more chat exports and save a single merged communication profile.

Running `setup` again regenerates the profile from scratch — the profile is always built from the full corpus you provide.

### Arguments

| Argument | Description |
|----------|-------------|
| `PATHS...` | One or more export files (`.txt` / `.json`), OR a directory containing exports |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--you`, `-u` | *(required)* | Your name as it appears in the exports |
| `--out`, `-o` | `data/profiles` | Directory to save the profile |

### Examples

Process an entire exports directory:

```bash
python -m cli.main setup data/exports/ --you "Rishit Babbar"
```

Process specific files:

```bash
python -m cli.main setup chat1.txt chat2.txt --you "Rishit Babbar"
```

Custom output directory:

```bash
python -m cli.main setup data/exports/ --you "Rishit Babbar" --out custom/profiles
```

---

## profile

**Description:** Display a saved communication profile with relationship breakdown, global style stats, and emoji usage.

### Arguments

| Argument | Description |
|----------|-------------|
| `YOU` | Name used when running `setup` |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dir`, `-d` | `data/profiles` | Directory where profiles are stored |

### Examples

```bash
python -m cli.main profile "Rishit Babbar"
```

```bash
python -m cli.main profile "Rishit Babbar" --dir custom/profiles
```

---

## parse

**Description:** Parse chat exports and display raw statistics (message counts, date range, sender breakdown). Does **not** save a profile — use `setup` for that.

### Arguments

| Argument | Description |
|----------|-------------|
| `PATHS...` | One or more export files, OR a directory containing exports |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--you`, `-u` | `""` | Highlight your messages in the breakdown |
| `--chat`, `-c` | `""` | Chat name override (only applies when a single file is provided) |

### Examples

```bash
python -m cli.main parse data/exports/chat.txt --you "Rishit Babbar"
```

```bash
python -m cli.main parse data/exports/ --you "Rishit Babbar"
```

```bash
python -m cli.main parse data/exports/chat.txt --you "Rishit Babbar" --chat "Work Group"
```

---

## signals

**Description:** Show the full relationship signal breakdown for every contact in an export.

Displays which specific words triggered each relationship category and the confidence score for each. Useful for validating and calibrating the classifier against real data.

### Arguments

| Argument | Description |
|----------|-------------|
| `FILE` | Chat export file to inspect (single file, not a directory) |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--you`, `-u` | *(required)* | Your sender name |
| `--contact`, `-c` | `""` | Show breakdown for one contact only |

### Examples

```bash
python -m cli.main signals "data/exports/chat.txt" --you "Rishit Babbar"
```

```bash
python -m cli.main signals "data/exports/chat.txt" --you "Rishit Babbar" --contact "Dhaval Sethi"
```

---

## summary

**Description:** Generate a natural-language summary of your communication profile using Ollama.

The summary is displayed in the terminal (not persisted to the profile JSON in the current version).

### Prerequisite

Start Ollama:

```bash
ollama serve
```

### Arguments

| Argument | Description |
|----------|-------------|
| `YOU` | Name used when running `setup` |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | `qwen3.5:9b` | Ollama model to use |
| `--dir`, `-d` | `data/profiles` | Directory where profiles are stored |

### Examples

```bash
python -m cli.main summary "Rishit Babbar"
```

```bash
python -m cli.main summary "Rishit Babbar" --model mistral
```

---

## store

**Description:** Embed chat exports into ChromaDB for semantic memory retrieval.

This populates the vector database used by `search` and `reply`. Only messages authored by `--you` are embedded.

### Arguments

| Argument | Description |
|----------|-------------|
| `PATHS...` | One or more export files, OR a directory containing exports |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--you`, `-u` | *(required)* | Your name — only YOUR messages get embedded |
| `--db` | `data/db/chroma` | ChromaDB storage path |
| `--clear` | `False` | Clear existing index before adding new messages |

### Examples

```bash
python -m cli.main store data/exports/ --you "Rishit Babbar"
```

```bash
python -m cli.main store data/exports/ --you "Rishit Babbar" --clear
```

```bash
python -m cli.main store data/exports/ --you "Rishit Babbar" --db custom/db/path
```

---

## search

**Description:** Search your past messages using semantic similarity.

Finds the most similar things you've said in the past to a given query. This is the same retrieval the `reply` command uses as context.

### Arguments

| Argument | Description |
|----------|-------------|
| `QUERY` | Text to search for in your past messages |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--you`, `-u` | *(required)* | Your sender name |
| `--top`, `-n` | `5` | Number of results to return |
| `--db` | `data/db/chroma` | ChromaDB storage path |
| `--chat`, `-c` | `""` | Filter results to a specific contact/chat |

### Examples

```bash
python -m cli.main search "what are you doing tonight" --you "Rishit Babbar"
```

```bash
python -m cli.main search "assignment deadline" --you "Rishit Babbar" --top 3
```

```bash
python -m cli.main search "plans for the weekend" --you "Rishit Babbar" --chat "Vibhu"
```

---

## reply

**Description:** Generate personalized reply drafts for incoming messages.

Uses RAG (semantic search over your past messages) combined with an Ollama model to draft replies that sound like you.

### Prerequisite

```bash
ollama serve
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INCOMING` | The message you received |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--you`, `-u` | *(required)* | Your name (matches profile) |
| `--contact`, `-c` | *(required)* | Who sent the message |
| `--num`, `-n` | `3` | Number of draft options to generate |
| `--model`, `-m` | `qwen3.5:9b` | Ollama model to use |
| `--db` | `data/db/chroma` | ChromaDB storage path |
| `--dir`, `-d` | `data/profiles` | Profiles directory |

### Examples

```bash
python -m cli.main reply "are you free tonight?" --you "Rishit Babbar" --contact "Priya"
```

```bash
python -m cli.main reply "bhai notes bhej de" --you "Rishit Babbar" --contact "Shreyansh" --num 5
```

```bash
python -m cli.main reply "what's the plan?" --you "Rishit Babbar" --contact "Vibhu" --model mistral
```

---

## check

**Description:** Verify environment setup, dependencies, and project directories.

Runs `scripts/setup_check.py` under the hood.

### Example

```bash
python -m cli.main check
```

---

# Utility Scripts

Standalone helper scripts located in:

```text
scripts/
```

---

## analyze_signals.py

Relationship signal analysis without the full CLI interface. Uses the same classifier as `setup` — there is exactly one source of truth for relationship scoring.

### Usage

```bash
python scripts/analyze_signals.py "data/exports/WhatsApp Chat with Shreyansh.txt" --you "Rishit Babbar"
```

```bash
python scripts/analyze_signals.py data/exports/ --you "Rishit Babbar"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--you`, `-u` | *(required)* | Your name as it appears in the exports |

---

## diagnose.py

Environment and import diagnostics. Checks Python version, venv, sys.path, all brain module imports, key dependencies, data directories, parser smoke test, and Ollama connectivity.

### Usage

```bash
python scripts/diagnose.py
```

---

## diagnose_reply.py

Debug reply generation and inspect raw Ollama output. Calls Ollama directly and prints the raw response with detailed annotations of what the parser sees.

### Usage

```bash
python scripts/diagnose_reply.py
```

```bash
python scripts/diagnose_reply.py --message "bhai free hai kal?"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--message`, `-m` | `"bhai free hai kal?"` | Test message to send |

---

## project_audit.py

Generate a structured snapshot of the PersonaOS codebase. Useful when providing project context to AI assistants.

### Usage

```bash
python scripts/project_audit.py
```

```bash
python scripts/project_audit.py --output project_state.txt
```

```bash
python scripts/project_audit.py --verbose
```

```bash
python scripts/project_audit.py --print
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `project_state.txt` | Output file path |
| `--verbose`, `-v` | `False` | Include module summaries for stub files |
| `--print`, `-p` | `False` | Also print the report to stdout |

---

## setup_check.py

Verify that the PersonaOS environment is correctly configured. Checks Python version, all dependencies, Ollama server, and data directories.

### Usage

```bash
python scripts/setup_check.py
```

---

# Quick Reference

| Command | Arguments | Purpose |
|---------|-----------|---------|
| `setup` | `PATHS...` | Build communication profile from exports |
| `profile` | `YOU` | View saved profile |
| `parse` | `PATHS...` | Parse exports without saving a profile |
| `signals` | `FILE` | View relationship signal breakdown |
| `summary` | `YOU` | Generate AI-powered profile summary |
| `store` | `PATHS...` | Embed messages into vector database |
| `search` | `QUERY` | Semantic memory search |
| `reply` | `INCOMING` | Generate personalized reply drafts |
| `check` | *(none)* | Validate environment |

## Common Options Reference

| Flag | Used by | Default | Description |
|------|---------|---------|-------------|
| `--you`, `-u` | setup, parse, signals, store, search, reply | *(varies)* | Your sender name |
| `--out`, `-o` | setup | `data/profiles` | Output directory for profiles |
| `--dir`, `-d` | profile, summary, reply | `data/profiles` | Profiles directory |
| `--model`, `-m` | summary, reply | `qwen3.5:9b` | Ollama model |
| `--db` | store, search, reply | `data/db/chroma` | ChromaDB path |
| `--contact`, `-c` | signals, search, reply | `""` | Filter by contact |
| `--chat`, `-c` | parse | `""` | Chat name override |
| `--num`, `-n` | reply | `3` / `5` | Number of results/drafts |
| `--top`, `-n` | search | `5` | Number of search results |
| `--clear` | store | `False` | Clear existing index first |

---

Generated automatically by the PersonaOS documentation generator.