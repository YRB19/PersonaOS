"""
scripts/project_audit.py

Generates a structured snapshot of the PersonaOS codebase.

Purpose:
    When you paste this output into Claude (or any AI assistant),
    it immediately knows the exact state of your project — what's
    implemented, what's a stub, what's broken, and what's installed.

Usage:
    python scripts/project_audit.py
    python scripts/project_audit.py --output project_state.txt
    python scripts/project_audit.py --verbose

Output file:  project_state.txt  (gitignored — update frequently)

Sections generated:
    TIMESTAMP        When the audit was run
    ENVIRONMENT      Python, venv, platform
    FILE INVENTORY   Every .py file with line count + status
    MODULE SUMMARIES Docstrings and TODOs per file
    IMPLEMENTATION   Phase-by-phase completion status
    DATA             What's in data/exports, data/db
    DEPENDENCIES     Installed packages (filtered to project deps)
    OLLAMA           Available local models
    GIT              Branch, last commit, dirty files
    TEST RESULTS     Last pytest run (if available)
    ERRORS           Import errors, broken modules
"""
import sys
import os
import subprocess
import importlib
import json
import ast
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse


# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = PROJECT_ROOT / "project_state.txt"

# Files we care about — in logical reading order
TRACKED_FILES = [
    # Phase 1 core
    "brain/__init__.py",
    "brain/profiler/__init__.py",
    "brain/profiler/models.py",
    "brain/profiler/parsers/__init__.py",
    "brain/profiler/parsers/whatsapp.py",
    "brain/profiler/parsers/telegram.py",
    "brain/profiler/analyzer.py",
    # Phase 2
    "brain/memory/__init__.py",
    "brain/memory/store.py",
    "brain/generator/__init__.py",
    "brain/generator/reply.py",
    # Infrastructure
    "cli/main.py",
    "scripts/setup_check.py",
    "scripts/project_audit.py",
    "tests/__init__.py",
    "tests/test_parsers.py",
]

# Key dependencies to check (subset of requirements.txt)
KEY_DEPS = [
    "pydantic", "dotenv", "rich", "typer", "ollama",
    "sentence_transformers", "chromadb", "emoji", "pytest",
]

# Phase → files mapping for completion tracking
PHASES = {
    "Phase 1 — Profiler": [
        "brain/profiler/models.py",
        "brain/profiler/parsers/whatsapp.py",
        "brain/profiler/parsers/telegram.py",
        "brain/profiler/analyzer.py",
    ],
    "Phase 2 — Memory": [
        "brain/memory/store.py",
    ],
    "Phase 2 — Generator": [
        "brain/generator/reply.py",
    ],
    "Infrastructure": [
        "cli/main.py",
        "tests/test_parsers.py",
    ],
}

# A file is a "stub" if it has fewer than this many non-blank, non-comment lines
STUB_LINE_THRESHOLD = 5


# ── Main ───────────────────────────────────────────────────────────────────────

def run_audit(verbose: bool = False, output: Optional[Path] = None) -> str:
    """Run the full audit and return the report as a string."""
    sections: list[str] = []

    def h(title: str) -> str:
        return f"\n===== {title} =====\n"

    # ── Timestamp + project root ───────────────────────────────────────────────
    sections.append(h("PersonaOS Project Audit"))
    sections.append(f"Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(f"Project root: {PROJECT_ROOT}")
    sections.append(f"Python:       {sys.version.split()[0]}")

    # ── Environment ────────────────────────────────────────────────────────────
    sections.append(h("ENVIRONMENT"))
    sections.append(_get_environment())

    # ── File inventory ─────────────────────────────────────────────────────────
    sections.append(h("FILE INVENTORY"))
    file_stats = _get_file_stats()
    sections.append(_format_file_inventory(file_stats))

    # ── Module summaries ───────────────────────────────────────────────────────
    sections.append(h("MODULE SUMMARIES"))
    sections.append(_get_module_summaries(file_stats, verbose=verbose))

    # ── Implementation status ──────────────────────────────────────────────────
    sections.append(h("IMPLEMENTATION STATUS"))
    sections.append(_get_implementation_status(file_stats))

    # ── Data directory ─────────────────────────────────────────────────────────
    sections.append(h("DATA DIRECTORY"))
    sections.append(_get_data_summary())

    # ── Dependencies ───────────────────────────────────────────────────────────
    sections.append(h("DEPENDENCIES"))
    sections.append(_get_dependencies())

    # ── Ollama ─────────────────────────────────────────────────────────────────
    sections.append(h("OLLAMA MODELS"))
    sections.append(_get_ollama_models())

    # ── Git ────────────────────────────────────────────────────────────────────
    sections.append(h("GIT STATUS"))
    sections.append(_get_git_status())

    # ── Test results ───────────────────────────────────────────────────────────
    sections.append(h("TEST STATUS"))
    sections.append(_get_test_status())

    # ── Import errors ──────────────────────────────────────────────────────────
    sections.append(h("IMPORT ERRORS"))
    sections.append(_check_imports())

    report = "\n".join(sections) + "\n"

    # Write to file
    out_path = output or OUTPUT_FILE
    out_path.write_text(report, encoding="utf-8")
    print(f"\n  Audit saved → {out_path.relative_to(PROJECT_ROOT)}\n")

    return report


# ── Section builders ───────────────────────────────────────────────────────────

def _get_environment() -> str:
    lines: list[str] = []

    # venv
    in_venv = sys.prefix != sys.base_prefix
    venv_path = Path(sys.prefix).relative_to(PROJECT_ROOT) if in_venv else None
    lines.append(f"venv active:  {'yes — ' + str(venv_path) if in_venv else 'NO — activate with: source .venv/bin/activate'}")

    # Platform
    import platform
    lines.append(f"Platform:     {platform.system()} {platform.release()} {platform.machine()}")

    # Python executable
    lines.append(f"Executable:   {sys.executable}")

    return "\n".join(lines)


def _get_file_stats() -> dict[str, dict]:
    """
    For each tracked file, return:
      exists, total_lines, code_lines, is_stub, docstring, todos, functions, classes
    """
    stats: dict[str, dict] = {}

    for rel_path in TRACKED_FILES:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            stats[rel_path] = {"exists": False}
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        code_lines = sum(
            1 for l in lines
            if l.strip() and not l.strip().startswith("#")
        )
        is_stub = code_lines < STUB_LINE_THRESHOLD

        # Parse AST for functions, classes, docstring
        docstring = ""
        functions: list[str] = []
        classes: list[str] = []
        todos: list[str] = []

        try:
            tree = ast.parse(text)
            # Module docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr)
                    and isinstance(tree.body[0].value, ast.Constant)):
                raw_doc = str(tree.body[0].value.value)
                # Take first 2 lines of docstring
                docstring = " | ".join(raw_doc.strip().splitlines()[:2])

            # Top-level functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except SyntaxError:
            docstring = "[SyntaxError — file unparseable]"

        # TODO / FIXME comments
        for i, line in enumerate(lines, 1):
            if any(marker in line.upper() for marker in ("TODO", "FIXME", "HACK", "XXX")):
                todos.append(f"  line {i:>3}: {line.strip()}")

        stats[rel_path] = {
            "exists": True,
            "total_lines": total,
            "code_lines": code_lines,
            "is_stub": is_stub,
            "docstring": docstring,
            "functions": functions[:10],   # cap at 10
            "classes": classes[:10],
            "todos": todos[:5],            # cap at 5
        }

    return stats


def _format_file_inventory(stats: dict[str, dict]) -> str:
    lines: list[str] = []
    current_dir = ""

    for rel_path, info in stats.items():
        # Print directory header when it changes
        dir_part = str(Path(rel_path).parent)
        if dir_part != current_dir:
            lines.append(f"\n{dir_part}/")
            current_dir = dir_part

        filename = Path(rel_path).name

        if not info["exists"]:
            lines.append(f"  ✗ {filename:<35}  [MISSING]")
            continue

        status = "[stub]" if info["is_stub"] else "[implemented]"
        lines.append(
            f"  {'○' if info['is_stub'] else '●'} {filename:<35}"
            f"  {info['total_lines']:>4} lines  {status}"
        )

    return "\n".join(lines)


def _get_module_summaries(stats: dict[str, dict], verbose: bool = False) -> str:
    lines: list[str] = []

    for rel_path, info in stats.items():
        if not info.get("exists"):
            continue
        if info.get("is_stub") and not verbose:
            continue  # skip stubs in non-verbose mode

        lines.append(f"\n{rel_path}")

        if info.get("docstring"):
            lines.append(f"  doc:     {info['docstring']}")

        if info.get("classes"):
            lines.append(f"  classes: {', '.join(info['classes'])}")

        if info.get("functions"):
            lines.append(f"  public:  {', '.join(info['functions'])}")

        if info.get("todos"):
            lines.append("  TODOs:")
            lines.extend(info["todos"])

    if not lines:
        lines.append("  (all implemented files have no docstrings yet)")

    return "\n".join(lines)


def _get_implementation_status(stats: dict[str, dict]) -> str:
    lines: list[str] = []

    for phase, files in PHASES.items():
        lines.append(f"\n{phase}:")
        all_done = True
        for f in files:
            info = stats.get(f, {})
            if not info.get("exists"):
                lines.append(f"  ✗ {f}  [MISSING]")
                all_done = False
            elif info.get("is_stub"):
                lines.append(f"  ○ {f}  [stub — {info['code_lines']} lines]")
                all_done = False
            else:
                lines.append(f"  ✓ {f}  [{info['total_lines']} lines]")
        if all_done:
            lines.append("  → Phase complete")

    return "\n".join(lines)


def _get_data_summary() -> str:
    lines: list[str] = []

    dirs = {
        "data/exports":  "Chat export files",
        "data/profiles": "Generated JSON profiles",
        "data/db":       "SQLite + ChromaDB",
    }

    for rel_dir, desc in dirs.items():
        path = PROJECT_ROOT / rel_dir
        if not path.exists():
            lines.append(f"  ✗ {rel_dir}/  [directory missing]")
            continue

        files = list(path.glob("*"))
        files = [f for f in files if f.name != ".gitkeep"]

        if not files:
            lines.append(f"  {rel_dir}/  empty  ({desc})")
        else:
            lines.append(f"  {rel_dir}/  {len(files)} file(s)  ({desc})")
            for f in files[:5]:
                size_kb = f.stat().st_size / 1024
                lines.append(f"    - {f.name}  ({size_kb:.1f} KB)")
            if len(files) > 5:
                lines.append(f"    ... and {len(files) - 5} more")

    return "\n".join(lines)


def _get_dependencies() -> str:
    lines: list[str] = []

    # pip freeze filtered to key deps
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=10
        )
        installed = {
            pkg["name"].lower().replace("-", "_"): pkg["version"]
            for pkg in json.loads(result.stdout)
        }
    except Exception as e:
        return f"  [could not read pip packages: {e}]"

    for dep in KEY_DEPS:
        key = dep.lower().replace("-", "_")
        if key in installed:
            lines.append(f"  ✓ {dep:<25} {installed[key]}")
        else:
            lines.append(f"  ✗ {dep:<25} NOT INSTALLED")

    return "\n".join(lines)


def _get_ollama_models() -> str:
    try:
        import ollama
        response = ollama.list()
        if not response or not response.models:
            return "  (no models found — is Ollama running?)"
        lines = []
        for m in response.models:
            size_gb = (m.size or 0) / 1e9
            lines.append(f"  {m.model:<40}  {size_gb:.1f} GB")
        return "\n".join(lines)
    except ImportError:
        return "  ollama Python client not installed"
    except Exception as e:
        return f"  Ollama not reachable: {e}"


def _get_git_status() -> str:
    lines: list[str] = []

    def git(cmd: list[str]) -> str:
        try:
            r = subprocess.run(
                ["git"] + cmd,
                cwd=PROJECT_ROOT,
                capture_output=True, text=True, timeout=5
            )
            return r.stdout.strip()
        except Exception:
            return ""

    branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
    if not branch:
        return "  git not initialised — run: git init && git add . && git commit -m 'init'"

    lines.append(f"  Branch:      {branch}")

    last_commit = git(["log", "-1", "--format=%h %s (%cr)"])
    if last_commit:
        lines.append(f"  Last commit: {last_commit}")

    status = git(["status", "--short"])
    if status:
        lines.append(f"  Dirty files:")
        for l in status.splitlines()[:10]:
            lines.append(f"    {l}")
    else:
        lines.append("  Working tree clean")

    return "\n".join(lines)


def _get_test_status() -> str:
    """Run pytest in --tb=no mode for a quick summary."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=60
        )
        # Take last 15 lines (summary)
        output_lines = (result.stdout + result.stderr).strip().splitlines()
        summary = output_lines[-15:] if len(output_lines) > 15 else output_lines
        return "\n".join(f"  {l}" for l in summary)
    except FileNotFoundError:
        return "  pytest not found — install with: pip install pytest"
    except Exception as e:
        return f"  Could not run tests: {e}"


def _check_imports() -> str:
    """Try to import each brain module and report failures."""
    modules_to_check = [
        "brain",
        "brain.profiler.models",
        "brain.profiler.parsers.whatsapp",
        "brain.profiler.parsers.telegram",
        "brain.profiler.analyzer",
        "brain.memory.store",
        "brain.generator.reply",
    ]

    # Add project root to sys.path for import
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    lines: list[str] = []
    any_error = False

    for mod in modules_to_check:
        try:
            importlib.import_module(mod)
            lines.append(f"  ✓ {mod}")
        except ImportError as e:
            lines.append(f"  ✗ {mod}  — ImportError: {e}")
            any_error = True
        except SyntaxError as e:
            lines.append(f"  ✗ {mod}  — SyntaxError: {e}")
            any_error = True
        except Exception as e:
            # Stub files may raise other errors on import
            lines.append(f"  ○ {mod}  — {type(e).__name__}: {e}")

    if not any_error:
        lines.append("\n  All modules importable.")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a PersonaOS project state snapshot."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=f"Output file path (default: {OUTPUT_FILE.relative_to(PROJECT_ROOT)})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include module summaries for stub files too"
    )
    parser.add_argument(
        "--print", "-p",
        action="store_true",
        help="Also print the report to stdout"
    )
    args = parser.parse_args()

    report = run_audit(verbose=args.verbose, output=args.output)

    if args.print:
        print(report)
    else:
        # Always print a compact summary to terminal
        print("\n── PersonaOS Project State ──\n")
        for line in report.splitlines():
            if any(line.startswith(s) for s in ("===", "  ✓", "  ✗", "  ○", "  →", "  Branch", "  Last")):
                print(line)
