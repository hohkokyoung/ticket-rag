"""
CLI for the ticket RAG system.

Usage:
  python main.py                          # interactive mode
  python main.py "ticket description"    # single query
  python main.py --retrieve-only "..."   # show retrieved tickets, no generation
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

console = Console()


def print_hits(hits: list[dict]):
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("#", width=3)
    table.add_column("Ticket ID", width=20)
    table.add_column("Category", width=18)
    table.add_column("Lang", width=6)
    table.add_column("Rerank Score", width=13)
    table.add_column("Preview", width=60)

    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        score = hit.get("rerank_score", hit.get("rrf_score", 0))
        preview = hit["text"][:120].replace("\n", " ") + "…"
        table.add_row(
            str(i),
            hit.get("ticket_id", ""),
            meta.get("category", ""),
            meta.get("language", "")[:5],
            f"{score:.4f}",
            preview,
        )

    console.print(table)


def run_query(pipeline, ticket_text: str, retrieve_only: bool = False, top_k: int = 5):
    console.print(f"\n[bold]Query:[/bold] {ticket_text}\n")

    with console.status("[bold green]Retrieving and reranking..."):
        if retrieve_only:
            hits = pipeline.retrieve_only(ticket_text, top_k=top_k)
            console.print(f"\n[bold cyan]Top {len(hits)} retrieved tickets:[/bold cyan]")
            print_hits(hits)
            for i, hit in enumerate(hits, 1):
                console.print(Panel(
                    hit["text"],
                    title=f"[bold]Ticket {i}: {hit.get('ticket_id', '')}[/bold]",
                    border_style="dim",
                ))
            return

        result = pipeline.query(ticket_text, top_k=top_k)

    console.print(f"[bold cyan]Retrieved {len(result.hits)} relevant tickets:[/bold cyan]")
    print_hits(result.hits)

    console.print(Panel(
        Markdown(result.answer),
        title="[bold green]Suggested Resolution[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


def interactive_mode(pipeline, top_k: int = 5):
    console.print(Panel(
        "[bold]Ticket RAG[/bold] — type a ticket description, [dim]'quit'[/dim] to exit, [dim]'?r <text>'[/dim] for retrieve-only",
        border_style="blue",
    ))

    while True:
        try:
            user_input = console.input("\n[bold blue]>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        retrieve_only = False
        if user_input.startswith("?r "):
            retrieve_only = True
            user_input = user_input[3:].strip()

        try:
            run_query(pipeline, user_input, retrieve_only=retrieve_only, top_k=top_k)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


def main():
    parser = argparse.ArgumentParser(description="Ticket RAG — find similar tickets and get AI-suggested resolutions")
    parser.add_argument("query", nargs="?", help="Ticket description (omit for interactive mode)")
    parser.add_argument("--retrieve-only", "-r", action="store_true", help="Show retrieved tickets without generation")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    args = parser.parse_args()

    with console.status("[bold green]Loading pipeline..."):
        try:
            from src.pipeline import build_pipeline
            pipeline = build_pipeline()
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

    console.print("[green]Pipeline ready.[/green]")

    if args.query:
        run_query(pipeline, args.query, retrieve_only=args.retrieve_only, top_k=args.top_k)
    else:
        interactive_mode(pipeline, top_k=args.top_k)


if __name__ == "__main__":
    main()
