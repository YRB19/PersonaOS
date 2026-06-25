"""
scripts/diagnose_reply.py

Run this to diagnose exactly why _parse_reply_json is returning empty.
It calls Ollama directly and prints the raw response with annotations.

Usage:
    python scripts/diagnose_reply.py
    python scripts/diagnose_reply.py --message "bhai free hai kal?"
"""
import sys
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def main(message: str) -> None:
    print("\n═══ PersonaOS Reply Diagnostic ═══\n")

    # ── 1. Test Ollama connection ──────────────────────────────────────────────
    section("1. Ollama connection")
    try:
        import ollama
        models = ollama.list()
        model_names = [m.model for m in models.models]
        print(f"  ✓ Ollama reachable — {len(model_names)} model(s): {', '.join(model_names[:3])}")
    except Exception as e:
        print(f"  ✗ Ollama not reachable: {e}")
        print("  Fix: ollama serve")
        return

    # ── 2. Test raw chat call ──────────────────────────────────────────────────
    section("2. Raw ollama.chat() response")
    model = "qwen3.5:9b"
    system = 'You help draft WhatsApp replies. Output ONLY this JSON: {"replies": ["draft1", "draft2", "draft3"]}'
    user   = f'Friend sent: "{message}"\n\nWrite 3 reply options as JSON.'

    print(f"  Model: {model}")
    print(f"  System: {system[:80]}...")
    print(f"  User:   {user[:80]}...")
    print("\n  Calling Ollama (may take 30–60s)...")

    try:
        response = ollama.generate(
            model=model,
            prompt=f"{system}\n\n---\n\n{user}",
            stream=False,
            think=False,
            options={"temperature": 0.85},
        )
        if isinstance(response, dict):
            raw = (response.get("response", "") or "").strip()
        else:
            raw = (getattr(response, "response", None) or "").strip()
            if not raw and hasattr(response, "__getitem__"):
                raw = (response.get("response", "") or "").strip()
        print(f"\n  ── RAW RESPONSE ({len(raw)} chars) ──")
        print(raw)
        print("  ── END OF RAW RESPONSE ──")
    except Exception as e:
        print(f"  ✗ Chat call failed: {e}")
        return

    # ── 3. Diagnose what's in the response ────────────────────────────────────
    section("3. Response analysis")

    has_think = "<think>" in raw
    has_json  = "{" in raw and "replies" in raw
    print(f"  Contains <think> blocks: {has_think}")
    print(f"  Contains JSON with 'replies': {has_json}")

    if has_think:
        think_content = re.findall(r"<think>(.*?)</think>", raw, re.DOTALL)
        print(f"  Think block length: {sum(len(t) for t in think_content)} chars")
        # Show what's left after stripping
        stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        print(f"\n  ── AFTER STRIPPING <think> ({len(stripped)} chars) ──")
        print(stripped)
        print("  ── END ──")

    # ── 4. Try parsing ────────────────────────────────────────────────────────
    section("4. Parsing result")
    from brain.generator.reply import _parse_reply_json, _strip_thinking
    cleaned = _strip_thinking(raw)
    result  = _parse_reply_json(raw, expected=3)

    print(f"  _strip_thinking output ({len(cleaned)} chars): {cleaned[:100]!r}...")
    print(f"  _parse_reply_json result: {result}")

    if result:
        print(f"\n  ✓ PARSING WORKS — got {len(result)} draft(s)")
        for i, r in enumerate(result, 1):
            print(f"    {i}. {r}")
    else:
        print("\n  ✗ PARSING FAILED — returned empty list")
        print("  The raw response format is not handled by any fallback layer.")
        print("  Copy the RAW RESPONSE above and share it so the parser can be fixed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", "-m", default="bhai free hai kal?")
    args = parser.parse_args()
    main(args.message)