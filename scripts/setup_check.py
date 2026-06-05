"""
Run this to confirm your PersonaOS environment is ready.
Usage: python scripts/setup_check.py
"""
import sys


def check(label: str, fn):
    try:
        result = fn()
        print(f"  ✓ {label}" + (f" — {result}" if result else ""))
    except Exception as e:
        print(f"  ✗ {label} — {e}")


print("\n── PersonaOS Environment Check ──\n")

check("Python version",
      lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

check("pydantic",
      lambda: __import__("pydantic").VERSION)

check("rich",
      lambda: __import__("rich").__version__)

check("typer",
      lambda: __import__("typer").__version__)

check("ollama client",
      lambda: __import__("ollama").__version__)

check("chromadb",
      lambda: __import__("chromadb").__version__)

check("sentence_transformers",
      lambda: __import__("sentence_transformers").__version__)

check("Ollama server reachable", lambda: (
    __import__("ollama").list() and "running"
))

check("PersonaOS brain importable",
      lambda: __import__("brain") and "ok")

print("\n── Data directories ──\n")
from pathlib import Path
for d in ["data/exports", "data/profiles", "data/db"]:
    exists = Path(d).exists()
    print(f"  {'✓' if exists else '✗'} {d}/")

print()