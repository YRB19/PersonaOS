"""
PersonaOS CLI — Phase 1 development tool.
Usage: python cli/main.py [COMMAND] [OPTIONS]

Commands:
    setup   Full onboarding pipeline: parse → analyze → save profile
    parse   Parse an export and show stats (no profile saved)
    profile Show a saved profile
    check   Verify environment and dependencies
"""
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer(
    help="PersonaOS — Communication Brain CLI",
    add_completion=False,
)
console = Console()


# ── setup ──────────────────────────────────────────────────────────────────────

@app.command()
def setup(
    file: str = typer.Argument(..., help="Path to WhatsApp .txt or Telegram .json export"),
    you: str = typer.Option(..., "--you", "-u", help="Your name as it appears in the export"),
    chat: str = typer.Option("", "--chat", "-c", help="Chat name override (defaults to filename)"),
    out: str = typer.Option("data/profiles", "--out", "-o", help="Directory to save profile"),
):
    """
    Full Phase 1 pipeline: parse export → build style profile → save to disk.

    This is the main command. Run it once per export file.

    Example:
        python cli/main.py setup data/exports/priya.txt --you "Rishit"
    """
    from brain.profiler.analyzer import build_profile, save_profile, profile_path
    from rich.progress import Progress, SpinnerColumn, TextColumn

    file_path = Path(file)
    if not file_path.exists():
        console.print(f"\n[red]✗[/red] File not found: {file}")
        console.print("  Put your export in [bold]data/exports/[/bold] and try again.\n")
        raise typer.Exit(1)

    ext = file_path.suffix.lower()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        # ── Step 1: Parse ──────────────────────────────────────────────────────
        task = progress.add_task("Parsing export...", total=None)

        try:
            if ext == ".txt":
                from brain.profiler.parsers.whatsapp import parse_whatsapp_export
                records = parse_whatsapp_export(file_path, chat_name=chat)
            elif ext == ".json":
                from brain.profiler.parsers.telegram import parse_telegram_export
                records = parse_telegram_export(file_path, chat_name=chat)
            else:
                console.print(f"\n[red]✗[/red] Unknown format: {ext}. Use .txt (WhatsApp) or .json (Telegram).\n")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]✗ Parse failed:[/red] {e}\n")
            raise typer.Exit(1)

        progress.update(task, description=f"Parsed {len(records)} messages. Building profile...")

        # ── Step 2: Build profile ──────────────────────────────────────────────
        try:
            profile = build_profile(records, user_name=you)
        except ValueError as e:
            console.print(f"\n[red]✗ Profile build failed:[/red] {e}\n")
            raise typer.Exit(1)

        progress.update(task, description="Saving profile...")

        # ── Step 3: Save ───────────────────────────────────────────────────────
        dest = save_profile(profile, profile_path(you, out))
        progress.update(task, description="Done.")

    # ── Print summary ──────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold green]✓ Profile built for {you}[/bold green]",
        subtitle=f"Saved → {dest}",
        style="green",
    ))

    _print_profile_summary(profile, you)


# ── profile ────────────────────────────────────────────────────────────────────

@app.command()
def profile(
    you: str = typer.Argument(..., help="Name used when running setup"),
    profiles_dir: str = typer.Option("data/profiles", "--dir", "-d"),
):
    """
    Display a saved communication profile.

    Example:
        python cli/main.py profile "Rishit"
    """
    from brain.profiler.analyzer import load_profile, profile_path

    p = profile_path(you, profiles_dir)
    try:
        prof = load_profile(p)
    except FileNotFoundError as e:
        console.print(f"\n[red]✗[/red] {e}\n")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(f"[bold]Communication Profile — {you}[/bold]", style="blue"))
    _print_profile_summary(prof, you)


# ── parse (raw stats, no profile saved) ───────────────────────────────────────

@app.command()
def parse(
    file: str = typer.Argument(..., help="Path to WhatsApp .txt or Telegram .json export"),
    you: str = typer.Option("", "--you", "-u", help="Highlight your messages (optional)"),
    chat: str = typer.Option("", "--chat", "-c", help="Chat name override"),
):
    """
    Parse a chat export and show statistics. Does not save a profile.

    Useful for exploring an export before committing to full setup.

    Example:
        python cli/main.py parse data/exports/priya.txt
    """
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"\n[red]✗[/red] File not found: {file}\n")
        raise typer.Exit(1)

    ext = file_path.suffix.lower()
    try:
        if ext == ".txt":
            from brain.profiler.parsers.whatsapp import parse_whatsapp_export
            records = parse_whatsapp_export(file_path, chat_name=chat)
        elif ext == ".json":
            from brain.profiler.parsers.telegram import parse_telegram_export
            records = parse_telegram_export(file_path, chat_name=chat)
        else:
            console.print(f"\n[red]✗[/red] Unknown format: {ext}\n")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Parse error:[/red] {e}\n")
        raise typer.Exit(1)

    if not records:
        console.print("\n[yellow]⚠ No messages parsed.[/yellow] Check the file format.\n")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(f"[bold]{file_path.name}[/bold] — {len(records)} messages", style="blue"))

    # Date range
    console.print(
        f"  Date range:   [cyan]{records[0].timestamp.date()}[/cyan]"
        f"  →  [cyan]{records[-1].timestamp.date()}[/cyan]"
    )

    # Message counts
    if you:
        your_msgs = [r for r in records if r.sender == you]
        console.print(f"  Your messages: [green]{len(your_msgs)}[/green] / {len(records)}")
    else:
        console.print(f"  Total:         [green]{len(records)}[/green]")

    # Media + emoji
    media_count = sum(1 for r in records if r.has_media)
    emoji_count = sum(1 for r in records if r.has_emoji)
    console.print(f"  With emoji:    {emoji_count}  ({emoji_count * 100 // len(records)}%)")
    console.print(f"  Media:         {media_count}  ({media_count * 100 // len(records)}%)")

    # Participants table
    senders: dict[str, int] = {}
    for r in records:
        senders[r.sender] = senders.get(r.sender, 0) + 1

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Sender", style="cyan")
    table.add_column("Messages", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("")

    for name, count in sorted(senders.items(), key=lambda x: -x[1]):
        pct = count * 100 // len(records)
        bar = "█" * (pct // 5)
        style = "green" if name == you else ""
        table.add_row(name, str(count), f"{pct}%", bar, style=style)

    console.print()
    console.print(table)


# ── check ──────────────────────────────────────────────────────────────────────

@app.command()
def check():
    """Verify environment, dependencies, and data directories."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/setup_check.py"],
        cwd=Path(__file__).parent.parent,
    )
    raise typer.Exit(result.returncode)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _print_profile_summary(profile, name: str):
    """Print a styled summary of a CommunicationProfile."""
    style = profile.global_style
    emoji = profile.global_emoji

    if not style:
        console.print("  [yellow]No style data — not enough messages.[/yellow]")
        return

    # ── Style table ────────────────────────────────────────────────────────────
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Field", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Messages analysed", str(profile.total_messages_analyzed))
    table.add_row("Avg message length", f"{style.avg_message_length:.0f} chars")
    table.add_row("Avg word count", f"{style.avg_word_count:.1f} words")
    table.add_row("Lowercase heavy", "yes" if style.uses_lowercase_heavily else "no")
    table.add_row("Uses punctuation", "yes" if style.uses_punctuation else "no")

    if style.common_openers:
        table.add_row("Common openers", "  ".join(style.common_openers[:5]))
    if style.slang_tokens:
        table.add_row("Slang used", "  ".join(style.slang_tokens[:6]))
    if style.typing_quirks:
        table.add_row("Typing quirks", " | ".join(style.typing_quirks))

    if emoji and emoji.total_emoji_count > 0:
        top = "  ".join(e for e, _ in emoji.top_emojis[:5])
        table.add_row("Top emoji", top)
        table.add_row("Emoji frequency", f"{emoji.emoji_frequency:.2f} per message")

    console.print(table)

    # ── Relationships ──────────────────────────────────────────────────────────
    if profile.relationships:
        console.print(f"\n  [dim]Relationships profiled:[/dim]")
        for rel in profile.relationships[:5]:
            console.print(f"    [cyan]{rel.contact_name:<25}[/cyan] {rel.message_count} messages")
    console.print()


if __name__ == "__main__":
    app()