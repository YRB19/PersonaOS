"""
PersonaOS CLI
Usage: python -m cli.main [COMMAND] [OPTIONS]

Commands:
    setup    Parse one or more exports (or a directory) and save a merged profile
    profile  Show a saved profile with relationship breakdown
    parse    Parse an export and show raw stats (no profile saved)
    signals  Show detailed relationship signal breakdown for each contact
    summary  Generate a natural language summary using Ollama
    check    Verify environment and dependencies
    suggest  Suggest emoji reactions and GIFs for an incoming message
    store    Embed your messages into ChromaDB for semantic memory retrieval
    search   Search your past messages by semantic similarity
    reply    Generate personalized reply drafts for an incoming message
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_export_files(paths: list[str]) -> list[Path]:
    """
    Accepts:
      - A single directory  → returns all .txt and .json files inside
      - One or more files   → returns those files directly
      - A glob pattern      → let typer/shell expand it (already in paths)

    Raises typer.Exit if nothing found.
    """
    resolved: list[Path] = []

    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            found = list(p.glob("*.txt")) + list(p.glob("*.json"))
            if not found:
                console.print(f"\n[yellow]⚠[/yellow]  No .txt or .json files found in {p}\n")
            resolved.extend(found)
        elif p.exists():
            resolved.append(p)
        else:
            console.print(f"\n[red]✗[/red]  File not found: {p}\n")

    if not resolved:
        console.print("[red]✗[/red]  No export files found. Check your paths.\n")
        raise typer.Exit(1)

    return resolved


def _parse_file(path: Path, chat_name: str = "") -> list:
    """Parse a single export file. Returns list of MessageRecord."""
    ext = path.suffix.lower()
    if ext == ".txt":
        from brain.profiler.parsers.whatsapp import parse_whatsapp_export
        return parse_whatsapp_export(path, chat_name=chat_name)
    elif ext == ".json":
        from brain.profiler.parsers.telegram import parse_telegram_export
        return parse_telegram_export(path, chat_name=chat_name)
    else:
        console.print(f"[yellow]⚠[/yellow]  Skipping {path.name} — unknown format ({ext})")
        return []


# ── setup ──────────────────────────────────────────────────────────────────────

@app.command()
def setup(
    paths: list[str] = typer.Argument(
        ...,
        help="One or more export files, OR a directory containing exports. "
             "Example: data/exports/ OR 'chat1.txt' 'chat2.txt'",
    ),
    you: str = typer.Option(..., "--you", "-u", help="Your name as it appears in the exports"),
    out: str = typer.Option("data/profiles", "--out", "-o", help="Directory to save profile"),
):
    """
    Parse one or more chat exports and save a single merged profile.

    Pass a directory to process all exports at once:
        python -m cli.main setup data/exports/ --you "Rishit Babbar"

    Or pass specific files:
        python -m cli.main setup chat1.txt chat2.txt --you "Rishit Babbar"

    Running setup again on the same (or more) files regenerates the profile
    from scratch. The profile is always built from the full corpus you provide.
    """
    from brain.profiler.analyzer import build_profile, save_profile, profile_path
    from rich.progress import Progress, SpinnerColumn, TextColumn

    export_files = _resolve_export_files(paths)

    console.print()
    console.print(Panel(
        f"[bold]Setting up profile for:[/bold] {you}\n"
        f"[dim]Processing {len(export_files)} file(s)[/dim]",
        style="blue",
    ))

    all_records = []
    sources = []
    failed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        # ── Step 1: Parse all files ────────────────────────────────────────────
        for export_file in export_files:
            task = progress.add_task(f"Parsing {export_file.name}...", total=None)
            try:
                records = _parse_file(export_file)
                if records:
                    all_records.extend(records)
                    sources.append(export_file.name)
                    progress.update(task, description=f"✓ {export_file.name} — {len(records)} messages")
                else:
                    progress.update(task, description=f"⚠ {export_file.name} — 0 messages (skipped)")
                    failed.append(export_file.name)
            except Exception as e:
                progress.update(task, description=f"✗ {export_file.name} — {e}")
                failed.append(export_file.name)

        if not all_records:
            console.print("\n[red]✗[/red]  No messages parsed from any file. Check export formats.\n")
            raise typer.Exit(1)

        # ── Step 2: Build merged profile ───────────────────────────────────────
        task = progress.add_task(
            f"Building profile from {len(all_records):,} messages...",
            total=None,
        )
        try:
            profile = build_profile(
                all_records,
                user_name=you,
                sources=sources,
            )
        except ValueError as e:
            console.print(f"\n[red]✗ Profile build failed:[/red] {e}\n")
            raise typer.Exit(1)

        # ── Step 3: Save ───────────────────────────────────────────────────────
        progress.update(task, description="Saving profile...")
        dest = save_profile(profile, profile_path(you, out))

    # ── Summary ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold green]✓ Profile saved for {you}[/bold green]\n"
        f"[dim]{len(all_records):,} messages · "
        f"{len(profile.relationships)} contacts · "
        f"{len(sources)} files merged[/dim]",
        subtitle=str(dest),
        style="green",
    ))

    if failed:
        console.print(f"[yellow]⚠[/yellow]  {len(failed)} file(s) skipped: {', '.join(failed)}")

    _print_profile_summary(profile, you)


# ── profile ────────────────────────────────────────────────────────────────────

@app.command()
def profile(
    you: str = typer.Argument(..., help="Name used when running setup"),
    profiles_dir: str = typer.Option("data/profiles", "--dir", "-d"),
):
    """
    Display a saved communication profile with relationship breakdown.

    Example:
        python -m cli.main profile "Rishit Babbar"
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


# ── parse ──────────────────────────────────────────────────────────────────────

@app.command()
def parse(
    file: str = typer.Argument(..., help="Path to WhatsApp .txt or Telegram .json export"),
    you: str = typer.Option("", "--you", "-u", help="Highlight your messages"),
    chat: str = typer.Option("", "--chat", "-c", help="Chat name override"),
):
    """
    Parse a single chat export and show raw statistics.
    Does not save a profile — use setup for that.

    Example:
        python -m cli.main parse data/exports/chat.txt --you "Rishit Babbar"
    """
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"\n[red]✗[/red]  File not found: {file}\n")
        raise typer.Exit(1)

    try:
        records = _parse_file(file_path, chat_name=chat)
    except Exception as e:
        console.print(f"\n[red]✗ Parse error:[/red] {e}\n")
        raise typer.Exit(1)

    if not records:
        console.print("\n[yellow]⚠[/yellow]  No messages parsed. Check the file format.\n")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(
        f"[bold]{file_path.name}[/bold]  —  {len(records):,} messages",
        style="blue",
    ))

    console.print(
        f"  Date range:  [cyan]{records[0].timestamp.date()}[/cyan]"
        f"  →  [cyan]{records[-1].timestamp.date()}[/cyan]"
    )

    if you:
        your_msgs = [r for r in records if r.sender == you]
        console.print(f"  Your msgs:   [green]{len(your_msgs):,}[/green] / {len(records):,}")

    media_count = sum(1 for r in records if r.has_media)
    emoji_count = sum(1 for r in records if r.has_emoji)
    console.print(f"  With emoji:  {emoji_count:,}  ({emoji_count * 100 // len(records)}%)")
    console.print(f"  Media:       {media_count:,}  ({media_count * 100 // len(records)}%)")

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
        table.add_row(name, f"{count:,}", f"{pct}%", bar, style=style)

    console.print()
    console.print(table)


# ── signals ────────────────────────────────────────────────────────────────────

@app.command()
def signals(
    file: str = typer.Argument(..., help="Chat export to inspect"),
    you: str = typer.Option(..., "--you", "-u", help="Your sender name"),
    contact: str = typer.Option("", "--contact", "-c",
                                help="Show breakdown for one contact only"),
):
    """
    Show the full relationship signal breakdown for every contact in an export.

    Displays which specific words triggered each relationship category
    and the confidence score for each. Useful for validating and
    calibrating the classifier against real data.

    Example:
        python -m cli.main signals "data/exports/chat.txt" --you "Rishit Babbar"
        python -m cli.main signals "data/exports/chat.txt" --you "Rishit Babbar" --contact "Dhaval Sethi"
    """
    from brain.profiler.classifier import get_all_scores
    from brain.profiler.analyzer import _group_by_contact

    file_path = Path(file)
    if not file_path.exists():
        console.print(f"\n[red]✗[/red]  File not found: {file}\n")
        raise typer.Exit(1)

    try:
        records = _parse_file(file_path)
    except Exception as e:
        console.print(f"\n[red]✗[/red]  Parse error: {e}\n")
        raise typer.Exit(1)

    if not records:
        console.print("\n[yellow]⚠[/yellow]  No messages parsed.\n")
        raise typer.Exit(1)

    contact_groups = _group_by_contact(records, you)

    if contact and contact not in contact_groups:
        console.print(
            f"\n[red]✗[/red]  Contact '{contact}' not found. "
            f"Available: {', '.join(contact_groups.keys())}\n"
        )
        raise typer.Exit(1)

    groups_to_show = (
        {contact: contact_groups[contact]}
        if contact else contact_groups
    )

    for contact_name, contact_records in groups_to_show.items():
        user_msgs = [r for r in contact_records if r.sender == you]
        if len(user_msgs) < 5:
            continue

        console.print()
        console.print(Panel(
            f"[bold]{contact_name}[/bold]  "
            f"[dim]{len(user_msgs)} messages from {you}[/dim]",
            style="blue",
        ))

        all_scores = get_all_scores(user_msgs, contact_records)

        if not all_scores:
            console.print("  [dim]Insufficient data for classification[/dim]")
            continue

        TYPE_COLOURS = {
            "close_friend": "green", "academic_peer": "blue",
            "family": "magenta", "colleague": "yellow",
            "romantic_interest": "red", "acquaintance": "dim",
        }

        for rel_type, (conf, rel_signals) in all_scores.items():
            colour = TYPE_COLOURS.get(rel_type, "white")
            bar = "█" * int(conf * 20)
            console.print(
                f"\n  [{colour}]{rel_type:<20}[/{colour}]  "
                f"[bold]{conf:.0%}[/bold]  [dim]{bar}[/dim]"
            )
            for sig in rel_signals:
                console.print(f"    [dim]→[/dim] {sig}")

        # Show winning classification
        best_type, best_conf = max(
            ((t, c) for t, (c, _) in all_scores.items()),
            key=lambda x: x[1],
        )
        console.print(
            f"\n  [bold]→ Classified as:[/bold] "
            f"[{TYPE_COLOURS.get(best_type, 'white')}]{best_type}[/{TYPE_COLOURS.get(best_type, 'white')}]"
            f"  ({best_conf:.0%} confidence)"
        )

    console.print()


# ── summary ────────────────────────────────────────────────────────────────────

@app.command()
def summary(
    you: str = typer.Argument(..., help="Name used when running setup"),
    model: str = typer.Option("qwen3.5:9b", "--model", "-m",
                              help="Ollama model to use"),
    profiles_dir: str = typer.Option("data/profiles", "--dir", "-d"),
):
    """
    Generate a natural language summary of your communication profile using Ollama.

    This is the first LLM step in PersonaOS. Ollama must be running:
        ollama serve

    The summary is saved back into the profile JSON.

    Example:
        python -m cli.main summary "Rishit Babbar"
        python -m cli.main summary "Rishit Babbar" --model mistral
    """
    from brain.profiler.analyzer import load_profile, profile_path
    from rich.progress import Progress, SpinnerColumn, TextColumn

    p = profile_path(you, profiles_dir)
    try:
        prof = load_profile(p)
    except FileNotFoundError as e:
        console.print(f"\n[red]✗[/red] {e}\n")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(
        f"[bold]Generating summary for {you}[/bold]\n"
        f"[dim]Model: {model}  |  Make sure Ollama is running: ollama serve[/dim]",
        style="blue",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Asking {model}...", total=None)
        try:
            # Import here to avoid import errors if Ollama not needed
            import ollama
            
            # Build prompt from profile
            style = prof.global_style
            prompt = f"""
Based on this communication profile, write a 2-3 sentence natural language summary:

Name: {you}
Total messages analyzed: {prof.total_messages_analyzed}
Number of relationships: {len(prof.relationships)}

Writing style:
- Average message length: {style.avg_message_length:.0f} characters
- Common openers: {', '.join(style.common_openers[:3])}
- Slang tokens: {', '.join(style.slang_tokens[:3])}
- Typing quirks: {', '.join(style.typing_quirks[:2]) if style.typing_quirks else 'none'}

Top 3 relationships:
{chr(10).join([f"- {r.contact_name}: {r.relationship_type} ({r.relationship_confidence:.0%} confidence)" for r in prof.relationships[:3]])}

Write a brief, human-readable summary of this person's communication style and key relationships.
            """.strip()
            
            response = ollama.generate(
                model=model,
                prompt=prompt,
                stream=False,
            )
            
            result = response.get("response", "").strip()
            progress.update(task, description="Done.")
            
            if not result:
                raise RuntimeError(f"Empty response from {model}")
        except Exception as e:
            console.print(f"\n[red]✗[/red] {e}\n")
            console.print("[dim]Make sure Ollama is running: ollama serve[/dim]\n")
            raise typer.Exit(1)

    console.print()
    console.print(Panel(
        f"[bold green]{result}[/bold green]",
        title="Communication Summary",
        style="green",
    ))
    console.print("[dim]  (Summary not persisted in this version)[/dim]\n")


# ── store ──────────────────────────────────────────────────────────────────────

@app.command()
def store(
    paths: list[str] = typer.Argument(
        ...,
        help="Export files or directory to embed. Same syntax as setup.",
    ),
    you: str = typer.Option(..., "--you", "-u", help="Your name — only YOUR messages get embedded"),
    db: str = typer.Option("data/db/chroma", "--db", help="ChromaDB storage path"),
    clear_first: bool = typer.Option(False, "--clear", help="Clear existing index before adding"),
):
    """
    Embed chat exports into ChromaDB for semantic memory retrieval.

    This is Phase 2A — run it after setup to populate the vector store.
    Only YOUR messages are embedded (--you flag). The store is used by
    the reply generator to find your most relevant past messages.

    Examples:
        python -m cli.main store data/exports/ --you "Rishit Babbar"
        python -m cli.main store data/exports/ --you "Rishit Babbar" --clear
    """
    from brain.memory.store import PersonaStore
    from rich.progress import Progress, SpinnerColumn, TextColumn

    export_files = _resolve_export_files(paths)

    console.print()
    console.print(Panel(
        f"[bold]Building memory store for:[/bold] {you}\n"
        f"[dim]{len(export_files)} file(s) → {db}[/dim]",
        style="blue",
    ))

    # ── Initialise store ───────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task("Loading embedding model...", total=None)

        try:
            persona_store = PersonaStore(chroma_path=db)
        except Exception as e:
            console.print(f"\n[red]✗[/red] Failed to initialise store: {e}\n")
            raise typer.Exit(1)

        if clear_first:
            progress.update(task, description="Clearing existing index...")
            persona_store.clear()
            console.print(f"  [dim]✓ Cleared existing index[/dim]")

        # ── Parse + embed all files ────────────────────────────────────────────
        total_stored = 0
        total_skipped = 0

        for export_file in export_files:
            progress.update(task, description=f"Embedding {export_file.name}...")
            try:
                records = _parse_file(export_file)
                if not records:
                    total_skipped += 1
                    continue
                n = persona_store.add_messages(records, sender_filter=you)
                total_stored += n
                progress.update(
                    task,
                    description=f"✓ {export_file.name} — {n} messages embedded",
                )
            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] {export_file.name}: {e}")
                total_skipped += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    stats = persona_store.stats()
    console.print()
    console.print(Panel(
        f"[bold green]✓ Memory store built[/bold green]\n"
        f"[dim]{total_stored:,} messages embedded this run · "
        f"{stats['total']:,} total in store · "
        f"senders: {', '.join(stats['senders'])}[/dim]",
        subtitle=db,
        style="green",
    ))

    if total_skipped:
        console.print(f"  [yellow]⚠[/yellow] {total_skipped} file(s) skipped\n")

    console.print(
        "  [dim]Run:[/dim] python -m cli.main search \"your query\" "
        f"--you \"{you}\" [dim]to test retrieval[/dim]\n"
    )


# ── search ─────────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Text to search for in your past messages"),
    you: str = typer.Option(..., "--you", "-u", help="Your sender name"),
    top: int = typer.Option(5, "--top", "-n", help="Number of results to return"),
    db: str = typer.Option("data/db/chroma", "--db", help="ChromaDB storage path"),
    chat: str = typer.Option("", "--chat", "-c", help="Filter to a specific contact/chat"),
):
    """
    Search your past messages by semantic similarity.

    This tests Phase 2A — finds the most similar things you've said
    in the past to a given query. This is exactly what the reply
    generator uses as context before drafting a reply.

    Examples:
        python -m cli.main search "what are you doing tonight" --you "Rishit Babbar"
        python -m cli.main search "assignment deadline" --you "Rishit Babbar" --top 3
        python -m cli.main search "plans for the weekend" --you "Rishit Babbar" --chat "Vibhu"
    """
    from brain.memory.store import PersonaStore

    try:
        persona_store = PersonaStore(chroma_path=db)
    except Exception as e:
        console.print(f"\n[red]✗[/red] Could not load store: {e}\n")
        raise typer.Exit(1)

    if persona_store.count() == 0:
        console.print(
            f"\n[yellow]⚠[/yellow] Store is empty. Run first:\n"
            f"  python -m cli.main store data/exports/ --you \"{you}\"\n"
        )
        raise typer.Exit(1)

    results = persona_store.retrieve_similar(
        text=query,
        top_k=top,
        sender_filter=you if you else None,
        chat_filter=chat if chat else None,
    )

    console.print()
    console.print(Panel(
        f"[bold]Search:[/bold] {query}\n"
        f"[dim]{persona_store.count():,} messages in store · showing top {top}[/dim]",
        style="blue",
    ))

    if not results:
        console.print("  [dim]No results found.[/dim]\n")
        return

    for i, r in enumerate(results, 1):
        sim_pct = f"{r['similarity']:.0%}"
        sim_bar = "█" * int(r["similarity"] * 15)
        chat_label = f"[dim]{r['chat_name']}[/dim]  " if r["chat_name"] else ""

        console.print(
            f"\n  [bold cyan]{i}.[/bold cyan]  "
            f"[green]{sim_pct}[/green]  [dim]{sim_bar}[/dim]  "
            f"{chat_label}"
        )
        console.print(f"     {r['content']}")

    console.print()


# ── reply ─────────────────────────────────────────────────────────────────────

@app.command()
def reply(
    incoming: str = typer.Argument(..., help="The message you received"),
    you: str      = typer.Option(..., "--you", "-u",      help="Your name (matches profile)"),
    contact: str  = typer.Option(..., "--contact", "-c",  help="Who sent the message"),
    num: int      = typer.Option(3,   "--num", "-n",      help="Number of draft options"),
    model: str    = typer.Option("qwen3.5:9b", "--model", "-m"),
    db: str       = typer.Option("data/db/chroma", "--db"),
    profiles_dir: str = typer.Option("data/profiles", "--dir", "-d"),
):
    """
    Generate personalized reply drafts for an incoming message.

    Requires Ollama running: ollama serve

    Examples:
        python -m cli.main reply "are you free tonight?" --you "Rishit Babbar" --contact "Priya"
        python -m cli.main reply "bhai notes bhej de" --you "Rishit Babbar" --contact "Shreyansh" --num 5
    """
    from brain.memory.store import PersonaStore
    from brain.generator.reply import ReplyGenerator
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console.print()
    incoming_preview = incoming[:60] + "..." if len(incoming) > 60 else incoming
    console.print(Panel(
        f"[bold]Generating reply drafts[/bold]\n"
        f"[dim]From: {contact}  ·  \"{incoming_preview}\"[/dim]",
        style="blue",
    ))

    # ── Load store ─────────────────────────────────────────────────────────────
    try:
        store      = PersonaStore(chroma_path=db)
        store_size = store.count()
    except Exception as e:
        console.print(f"\n[red]✗[/red] Could not load store: {e}\n")
        raise typer.Exit(1)

    if store_size == 0:
        console.print(
            f"\n[yellow]⚠[/yellow]  Vector store is empty — RAG will be skipped.\n"
            f"  [dim]Run: python -m cli.main store data/exports/ --you \"{you}\"[/dim]\n"
        )

    gen = ReplyGenerator(store, profiles_dir=profiles_dir, model=model)

    # ── Generate ───────────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task(f"Asking {model} to write like you...", total=None)
        try:
            drafts = gen.generate(
                incoming=incoming,
                from_contact=contact,
                user_name=you,
                num_drafts=num,
            )
            progress.update(task, description="Done.")
        except RuntimeError as e:
            console.print(f"\n[red]✗[/red] {e}\n")
            raise typer.Exit(1)

    # ── Display ────────────────────────────────────────────────────────────────
    if not drafts:
        console.print(
            "\n[yellow]⚠[/yellow]  No drafts generated.\n"
            "  Check that Ollama is running and qwen3.5:9b is pulled.\n"
        )
        raise typer.Exit(1)

    console.print(f"\n  [dim]{store_size:,} messages in memory · {len(drafts)} drafts · {model}[/dim]\n")

    for i, draft in enumerate(drafts, 1):
        style = "cyan" if i == 1 else "dim"
        console.print(Panel(
            f"[white]{draft}[/white]",
            title=f"[bold {style}]Draft {i}[/bold {style}]",
            border_style=style,
        ))

    console.print(
        "\n  [dim]Pick one, edit it, send it yourself. You're in control.[/dim]\n"
    )


# ── suggest ────────────────────────────────────────────────────────────────────

@app.command()
def suggest(
    incoming: str = typer.Argument(..., help="The incoming message to get suggestions for"),
    you: str      = typer.Option(..., "--you", "-u", help="Your name (matches profile)"),
    contact: str  = typer.Option(..., "--contact", "-c", help="Who sent the message"),
    type: str     = typer.Option(
        "all", "--type", "-t",
        help="What to suggest: emoji | gif | all  (default: all)"
    ),
    profiles_dir: str = typer.Option("data/profiles", "--dir", "-d"),
):
    """
    Suggest reaction emojis and GIFs for an incoming message.

    Emoji suggestions are fully local (no API needed).
    GIF suggestions require TENOR_API_KEY in .env (free, 2 min setup).

    Examples:
        python -m cli.main suggest "lmao bhai 💀" --you "Rishit Babbar" --contact "Shreyansh"
        python -m cli.main suggest "you free tonight?" --you "Rishit Babbar" --contact "Priya" --type emoji
        python -m cli.main suggest "got the placement!!" --you "Rishit Babbar" --contact "Dhaval Sethi" --type gif
    """
    from brain.media.emoji_suggester import suggest_reactions
    from brain.media.gif_suggester import search_gifs, build_gif_query
    from brain.profiler.analyzer import load_profile, profile_path

    # ── Load profile + relationship (best-effort) ──────────────────────────────
    prof = None
    relationship = None
    try:
        prof = load_profile(profile_path(you, profiles_dir))
        if prof:
            contact_l = contact.lower()
            for rel in prof.relationships:
                if (rel.contact_name.lower() == contact_l
                        or contact_l in rel.contact_name.lower()):
                    relationship = rel
                    break
    except Exception:
        pass

    incoming_preview = incoming[:55] + "..." if len(incoming) > 55 else incoming
    console.print()
    console.print(Panel(
        f"[bold]Media Suggestions[/bold]\n"
        f"[dim]Message: \"{incoming_preview}\"  ·  From: {contact}[/dim]",
        style="blue",
    ))

    rel_label = relationship.relationship_type if relationship else "unknown"
    console.print(f"  [dim]Relationship: {rel_label}  ·  "
                  f"Profile: {'loaded' if prof else 'not found (generic suggestions)'}[/dim]\n")

    # ── Emoji / reaction suggestions ───────────────────────────────────────────
    if type in ("emoji", "all"):
        reactions = suggest_reactions(
            incoming,
            profile=prof,
            relationship=relationship,
            top_k=8,
        )

        if reactions:
            emoji_row = "  ".join(r.emoji for r in reactions[:8])
            console.print(f"  [bold]React with:[/bold]  {emoji_row}\n")

            table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            table.add_column("Emoji", style="bold", min_width=4)
            table.add_column("Confidence", justify="right", min_width=6)
            table.add_column("Reason", style="dim")

            for r in reactions[:5]:
                table.add_row(r.emoji, f"{r.confidence:.0%}", r.reason)

            console.print(table)
        else:
            console.print("  [dim]No emoji suggestions.[/dim]")

    # ── GIF suggestions ───────────────────────────────────────────────────────────
    if type in ("gif", "all"):
        import os
        has_key = bool(os.getenv("TENOR_API_KEY", "").strip())
        gif_query = build_gif_query(incoming, relationship)

        console.print(f"\n  [bold]GIF search query:[/bold]  [cyan]{gif_query}[/cyan]")

        if has_key:
            gifs = search_gifs(incoming, relationship=relationship)
            if gifs:
                console.print(f"  [dim]{len(gifs)} result(s) from Tenor:[/dim]\n")
                for i, gif in enumerate(gifs, 1):
                    console.print(f"  [bold cyan]{i}.[/bold cyan]  {gif.title}")
                    console.print(f"     [dim]{gif.url}[/dim]")
            else:
                console.print("  [yellow]⚠[/yellow]  No GIFs returned (check TENOR_API_KEY or try a different query)")
        else:
            console.print(
                "  [dim]Add TENOR_API_KEY to .env for live GIF search.[/dim]\n"
                "  [dim]Get a free key: https://developers.google.com/tenor/guides/quickstart[/dim]"
            )

    console.print()


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


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_profile_summary(profile, name: str):
    """Print a styled summary of a CommunicationProfile."""

    # ── Sources ────────────────────────────────────────────────────────────────
    if profile.sources_analyzed:
        console.print(f"\n  [dim]Built from:[/dim] {', '.join(profile.sources_analyzed)}")

    # ── Global style ───────────────────────────────────────────────────────────
    style = profile.global_style
    emoji = profile.global_emoji

    if not style:
        console.print("  [yellow]No style data — not enough messages.[/yellow]")
        return

    console.print(f"\n  [bold]Global Style[/bold]  [dim]({profile.total_messages_analyzed:,} messages)[/dim]")

    stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats.add_column("Field", style="dim")
    stats.add_column("Value", style="bold")

    stats.add_row("Avg message length", f"{style.avg_message_length:.0f} chars")
    stats.add_row("Avg word count", f"{style.avg_word_count:.1f} words")
    stats.add_row("Lowercase heavy", "yes" if style.uses_lowercase_heavily else "no")
    stats.add_row("Uses punctuation", "yes" if style.uses_punctuation else "no")
    if style.common_openers:
        stats.add_row("Common openers", "  ".join(style.common_openers[:5]))
    if style.slang_tokens:
        stats.add_row("Slang", "  ".join(style.slang_tokens[:6]))
    if style.typing_quirks:
        stats.add_row("Quirks", " · ".join(style.typing_quirks))
    if emoji and emoji.total_emoji_count > 0:
        top = "  ".join(e for e, _ in emoji.top_emojis[:5])
        stats.add_row("Top emoji", top)
        stats.add_row("Emoji / message", f"{emoji.emoji_frequency:.2f}")

    console.print(stats)

    # ── Relationships ──────────────────────────────────────────────────────────
    if not profile.relationships:
        console.print("  [dim]No relationships profiled.[/dim]\n")
        return

    console.print(f"\n  [bold]Relationships[/bold]  [dim]({len(profile.relationships)} contacts)[/dim]\n")

    rel_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    rel_table.add_column("Contact", style="cyan", min_width=22)
    rel_table.add_column("Type", min_width=18)
    rel_table.add_column("Confidence", justify="right", min_width=10)
    rel_table.add_column("Messages", justify="right")
    rel_table.add_column("Top signal", style="dim")

    TYPE_COLOURS = {
        "close_friend":      "green",
        "academic_peer":     "blue",
        "family":            "magenta",
        "colleague":         "yellow",
        "romantic_interest": "red",
        "acquaintance":      "dim",
        "unknown":           "dim",
    }

    for rel in profile.relationships:
        colour = TYPE_COLOURS.get(rel.relationship_type, "white")
        type_label = f"[{colour}]{rel.relationship_type}[/{colour}]"
        confidence_pct = f"{rel.relationship_confidence * 100:.0f}%"
        top_signal = rel.relationship_signals[0] if rel.relationship_signals else "—"
        # Truncate signal if too long
        if len(top_signal) > 42:
            top_signal = top_signal[:39] + "..."
        rel_table.add_row(
            rel.contact_name[:22],
            type_label,
            confidence_pct,
            f"{rel.message_count:,}",
            top_signal,
        )

    console.print(rel_table)
    console.print()


if __name__ == "__main__":
    app()