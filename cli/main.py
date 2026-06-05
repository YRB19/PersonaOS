"""
PersonaOS CLI — Phase 1 development tool.
Usage: python cli/main.py
"""
import typer
from rich.console import Console
from rich.panel import Panel
from pathlib import Path

app = typer.Typer(help="PersonaOS Communication Brain")
console = Console()


@app.command()
def parse(
    file: str = typer.Argument(..., help="Path to WhatsApp export .txt file"),
    chat: str = typer.Option("", help="Chat name (defaults to filename)"),
    you: str = typer.Option(..., help="Your name as it appears in the export"),
):
    """Parse a WhatsApp chat export and show basic stats."""
    from brain.profiler.parsers.whatsapp import parse_whatsapp_export

    console.print(Panel(f"[bold]Parsing:[/bold] {file}", style="blue"))

    records = parse_whatsapp_export(file, chat_name=chat)
    your_messages = [r for r in records if r.sender == you]

    console.print(f"\n[green]Total messages:[/green] {len(records)}")
    console.print(f"[green]Your messages:[/green]  {len(your_messages)}")
    console.print(f"[green]Date range:[/green]     {records[0].timestamp.date()} → {records[-1].timestamp.date()}")

    senders = {}
    for r in records:
        senders[r.sender] = senders.get(r.sender, 0) + 1
    console.print("\n[bold]Participants:[/bold]")
    for name, count in sorted(senders.items(), key=lambda x: -x[1]):
        bar = "█" * min(30, count // max(1, len(records) // 30))
        console.print(f"  {name:<25} {count:>5}  {bar}")


@app.command()
def check():
    """Verify environment and dependencies."""
    import subprocess
    subprocess.run(["python", "scripts/setup_check.py"])


if __name__ == "__main__":
    app()