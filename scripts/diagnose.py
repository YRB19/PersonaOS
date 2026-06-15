#!/usr/bin/env python
"""
scripts/diagnose.py

PersonaOS environment and import diagnostic.
Run this when something breaks and you're not sure why.

Usage:
    python scripts/diagnose.py

What it checks:
    1. Python version and venv
    2. sys.path (can brain.* be found?)
    3. All brain module imports
    4. Key dependencies
    5. data/ directory contents
    6. Ollama connectivity
    7. Suggests next command to run
"""
import sys
from pathlib import Path

# ── CRITICAL: Add project root to sys.path ─────────────────────────────────────
# Without this, `python scripts/diagnose.py` adds scripts/ to sys.path,
# not the project root, and all `brain.*` imports fail immediately.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Helpers ────────────────────────────────────────────────────────────────────

def ok(label, detail=""):
    detail_str = f"  — {detail}" if detail else ""
    print(f"  ✓  {label}{detail_str}")

def fail(label, detail=""):
    detail_str = f"  — {detail}" if detail else ""
    print(f"  ✗  {label}{detail_str}")

def warn(label, detail=""):
    detail_str = f"  — {detail}" if detail else ""
    print(f"  ⚠  {label}{detail_str}")

def section(title):
    print(f"\n── {title} {'─' * (50 - len(title))}")


print("=== PersonaOS Diagnostic ===")
print(f"    Project root: {PROJECT_ROOT}")
print(f"    Python:       {sys.version.split()[0]}")


# ── 1. venv ────────────────────────────────────────────────────────────────────
section("Virtual Environment")

in_venv = sys.prefix != sys.base_prefix
if in_venv:
    venv_path = Path(sys.prefix).relative_to(PROJECT_ROOT)
    ok("venv active", str(venv_path))
else:
    fail("venv NOT active", "run: source .venv/bin/activate")


# ── 2. sys.path ────────────────────────────────────────────────────────────────
section("sys.path")

if str(PROJECT_ROOT) in sys.path:
    ok("Project root in sys.path", str(PROJECT_ROOT))
else:
    fail("Project root NOT in sys.path", "brain.* imports will fail")


# ── 3. Brain module imports ────────────────────────────────────────────────────
section("Brain Module Imports")

modules = [
    ("brain",                              "brain package"),
    ("brain.profiler.models",              "MessageRecord, CommunicationProfile"),
    ("brain.profiler.parsers.whatsapp",    "parse_whatsapp_export"),
    ("brain.profiler.parsers.telegram",    "parse_telegram_export"),
    ("brain.profiler.analyzer",            "build_profile, save_profile, generate_summary"),
    ("brain.profiler.classifier",          "classify_relationship, get_all_scores"),
    ("brain.memory.store",                 "stub — ok if empty"),
    ("brain.generator.reply",              "stub — ok if empty"),
]

import importlib
all_imports_ok = True
for mod_path, description in modules:
    try:
        importlib.import_module(mod_path)
        ok(mod_path, description)
    except ImportError as e:
        fail(mod_path, f"ImportError: {e}")
        all_imports_ok = False
    except Exception as e:
        warn(mod_path, f"{type(e).__name__}: {e}")


# ── 4. Key dependencies ────────────────────────────────────────────────────────
section("Dependencies")

deps = [
    ("pydantic",              "data models"),
    ("dotenv",                "python-dotenv — .env loading"),
    ("rich",                  "CLI formatting"),
    ("typer",                 "CLI framework"),
    ("ollama",                "local LLM client"),
    ("sentence_transformers", "embeddings (Phase 2)"),
    ("chromadb",              "vector store (Phase 2)"),
    ("emoji",                 "emoji detection"),
    ("pytest",                "testing"),
]

for pkg, desc in deps:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "?"))
        ok(pkg, f"{version}  ({desc})")
    except ImportError:
        if pkg == "dotenv":
            fail(pkg, "MISSING — run: pip install python-dotenv")
        else:
            fail(pkg, f"MISSING — run: pip install {pkg}")


# ── 5. Data directories ────────────────────────────────────────────────────────
section("Data Directories")

data_dirs = {
    "data/exports":  "chat export files",
    "data/profiles": "saved user profiles",
    "data/db":       "ChromaDB + SQLite",
}

for rel_dir, desc in data_dirs.items():
    path = PROJECT_ROOT / rel_dir
    if not path.exists():
        fail(rel_dir, "directory missing — run setup")
        continue
    files = [f for f in path.iterdir() if f.name != ".gitkeep"]
    if files:
        ok(rel_dir, f"{len(files)} file(s)  ({desc})")
        for f in sorted(files)[:3]:
            size_kb = f.stat().st_size / 1024
            print(f"       {f.name}  ({size_kb:.1f} KB)")
        if len(files) > 3:
            print(f"       ... and {len(files) - 3} more")
    else:
        warn(rel_dir, f"empty  ({desc})")


# ── 6. Parse test ──────────────────────────────────────────────────────────────
section("Parser Smoke Test")

exports_dir = PROJECT_ROOT / "data/exports"
txt_files = list(exports_dir.glob("*.txt")) if exports_dir.exists() else []

if not txt_files:
    warn("No .txt files in data/exports/", "export a WhatsApp chat first")
else:
    test_file = txt_files[0]
    try:
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        records = parse_whatsapp_export(test_file)
        if records:
            ok(
                f"Parsed {test_file.name}",
                f"{len(records)} messages  |  "
                f"first: {records[0].sender} @ {records[0].timestamp.date()}"
            )
        else:
            fail(f"Parsed {test_file.name}", "0 messages — check format")
    except Exception as e:
        fail(f"Parse failed on {test_file.name}", str(e))


# ── 7. Ollama ──────────────────────────────────────────────────────────────────
section("Ollama")

try:
    import ollama
    models = ollama.list()
    model_names = [m.model for m in models.models] if models.models else []
    if model_names:
        ok("Ollama reachable", f"{len(model_names)} model(s): {', '.join(model_names[:3])}")
    else:
        warn("Ollama reachable but no models", "run: ollama pull qwen3.5:9b")
except Exception as e:
    warn("Ollama not reachable", "start it: ollama serve")


# ── Summary ────────────────────────────────────────────────────────────────────
section("Suggested Next Step")

profiles_dir = PROJECT_ROOT / "data/profiles"
profiles = list(profiles_dir.glob("*.json")) if profiles_dir.exists() else []

if not all_imports_ok:
    print("  Fix import errors above first.")
elif not txt_files:
    print("  Add a WhatsApp export to data/exports/ and run:")
    print('  python -m cli.main setup data/exports/ --you "Your Name"')
elif not profiles:
    print("  Run setup to build your first profile:")
    print('  python -m cli.main setup data/exports/ --you "Rishit Babbar"')
else:
    print("  Everything looks good. Profile exists. Try:")
    print('  python -m cli.main profile "Rishit Babbar"')
    print('  python -m cli.main summary "Rishit Babbar"  # needs ollama serve')

print()