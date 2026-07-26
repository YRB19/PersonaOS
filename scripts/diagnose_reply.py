"""
scripts/diagnose_reply.py

Calls the real build_prompt() from brain/generator/reply.py and the exact
same Ollama call path used in production.  Prints the raw LLM response
annotated with parsing diagnostics.

Usage:
    python scripts/diagnose_reply.py
    python scripts/diagnose_reply.py --message "bhai free hai kal?"
    python scripts/diagnose_reply.py --message "kya kar rahe ho?" --contact "Vibhu"
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


def build_system_user(contact: str, message: str) -> tuple[str, str]:
    """
    Call the real build_prompt() from the production pipeline.
    When no profile / store is available the prompt degrades gracefully —
    it still produces valid system + user strings using defaults.
    """
    from brain.generator.reply import build_prompt
    return build_prompt(
        incoming=message,
        from_contact=contact,
        profile=None,
        relationship=None,
        examples=[],
    )


def main(message: str, contact: str) -> None:
    print("\n═══ PersonaOS Reply Diagnostic ═══\n")

    # ── 1. Ollama connection ──────────────────────────────────────────────────
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

    # ── 2. Build prompt via the real production pipeline ──────────────────────
    section("2. Prompt (from build_prompt())")
    model = "qwen3.5:9b"

    system, user = build_system_user(contact, message)

    print(f"  Model: {model}")
    print(f"  Contact: {contact}")
    print(f"  System prompt ({len(system)} chars):")
    print(f"\n{system}\n")
    print(f"  User prompt ({len(user)} chars):")
    print(f"\n{user}\n")
    print("\n  Calling Ollama (may take 30–60s)...")

    # ── 3. Call Ollama (same code path as production _call_ollama) ───────────
    section("3. Raw ollama.generate() response")
    combined = f"{system}\n\n---\n\n{user}"

    try:
        try:
            response = ollama.generate(
                model=model,
                prompt=combined,
                stream=False,
                think=False,
                options={"temperature": 0.85},
            )
        except TypeError:
            # Older ollama library versions don't support think= kwarg
            response = ollama.generate(
                model=model,
                prompt=combined,
                stream=False,
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
        print(f"  ✗ Ollama call failed: {e}")
        return

    # ── 4. Response analysis ──────────────────────────────────────────────────
    section("4. Response analysis")

    has_think = "<think>" in raw
    has_json  = "{" in raw and "replies" in raw
    print(f"  Contains <think> blocks: {has_think}")
    print(f"  Contains JSON with 'replies': {has_json}")

    if has_think:
        think_content = re.findall(r"<think>(.*?)</think>", raw, re.DOTALL)
        print(f"  Think block length: {sum(len(t) for t in think_content)} chars")
        stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        print(f"\n  ── AFTER STRIPPING <think> ({len(stripped)} chars) ──")
        print(stripped)
        print("  ── END ──")

    # ── 5. Try parsing ───────────────────────────────────────────────────────
    section("5. Parsing result")
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
    parser.add_argument("--contact", "-c", default="Friend",
                        help="Contact / sender name (default: Friend)")
    args = parser.parse_args()
    main(args.message, args.contact)