"""
Train a lightweight category classifier on top of frozen embeddings.

The embedding model (multilingual-e5-large) stays untouched — we just fit a
LogisticRegression on the 768-dim vectors it produces. Trains in seconds, no GPU needed.

At query time the pipeline calls predict_category(query_embedding) to get the most
likely support category, then filters ChromaDB + BM25 to that bucket before searching.
This directly fixes the root cause of Category Precision@1 = 64%: without filtering,
25% of the search space is noise from other categories.

Usage:
    python scripts/train_classifier.py                # 500 tickets per category (fast)
    python scripts/train_classifier.py --all          # all training tickets (slow, ~5 min)
    python scripts/train_classifier.py --n-per-cat 200
"""

import sys
import os
import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

_CLASSIFIER_PATH = Path(".category_classifier.pkl")


def main():
    parser = argparse.ArgumentParser(description="Train category classifier for ticket RAG")
    parser.add_argument("--n-per-cat", type=int, default=500,
                        help="Max tickets to embed per category (default: 500)")
    parser.add_argument("--all", action="store_true",
                        help="Use all training tickets (ignores --n-per-cat)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Load training tickets ──────────────────────────────────────────────
    console.print("[bold]Step 1/4[/bold] Loading training tickets...")
    from src.ingestion.loader import load_tickets

    meta_path = Path(".holdout_meta.json")
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        data_dir = meta["train_dir"]
        console.print(f"  Using holdout train split: [cyan]{data_dir}[/cyan]")
    else:
        data_dir = "data"
        console.print(f"  No holdout split found — using: [cyan]{data_dir}[/cyan]")

    tickets = load_tickets(data_dir)
    tickets = [t for t in tickets if t.get("category", "").strip()]

    # Group by category
    by_category: dict[str, list] = defaultdict(list)
    for t in tickets:
        by_category[t["category"]].append(t)

    categories = sorted(by_category.keys())
    console.print(f"  Found [bold]{len(tickets)}[/bold] tickets across "
                  f"[bold]{len(categories)}[/bold] categories: {', '.join(categories)}")

    # ── 2. Sample ─────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 2/4[/bold] Sampling tickets...")
    sampled = []
    for cat, cat_tickets in by_category.items():
        if args.all:
            chosen = cat_tickets
        else:
            chosen = random.sample(cat_tickets, min(args.n_per_cat, len(cat_tickets)))
        sampled.extend(chosen)
        console.print(f"  {cat}: [cyan]{len(chosen)}[/cyan] tickets")

    random.shuffle(sampled)
    console.print(f"  Total: [bold]{len(sampled)}[/bold] tickets to embed")

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 3/4[/bold] Embedding tickets...")
    from src.ingestion.embedder import get_embedder
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    embedder = get_embedder()

    texts = [f"{t.get('subject', '')} {t.get('body', '')}".strip() for t in sampled]
    labels = [t["category"] for t in sampled]

    # Embed in batches of 256
    batch_size = 256
    all_embeddings = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        task = progress.add_task("Embedding...", total=len(texts))
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = embedder.embed_passages(batch)
            all_embeddings.extend(embs)
            progress.advance(task, len(batch))

    import numpy as np
    X = np.array(all_embeddings)
    y = labels
    console.print(f"  Embedding matrix: [cyan]{X.shape}[/cyan]")

    # ── 4. Train + evaluate ───────────────────────────────────────────────────
    console.print("\n[bold]Step 4/4[/bold] Training classifier...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)

    # 5-fold cross-validation to get honest accuracy
    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
    console.print(f"  5-fold CV accuracy: [bold green]{cv_scores.mean():.1%}[/bold green] "
                  f"(±{cv_scores.std():.1%})")

    # Fit on full sample
    clf.fit(X, y_enc)

    # Per-class accuracy table
    from sklearn.metrics import classification_report
    y_pred = clf.predict(X)
    report = classification_report(y_enc, y_pred, target_names=le.classes_, output_dict=True)

    t = Table(title="Per-category accuracy (train set)", box=box.SIMPLE_HEAVY,
              header_style="bold cyan")
    t.add_column("Category", style="bold")
    t.add_column("Precision", justify="right")
    t.add_column("Recall", justify="right")
    t.add_column("F1", justify="right")
    t.add_column("Support", justify="right")

    for cat in le.classes_:
        r = report[cat]
        t.add_row(
            cat,
            f"{r['precision']:.1%}",
            f"{r['recall']:.1%}",
            f"{r['f1-score']:.1%}",
            str(int(r['support'])),
        )
    console.print(t)

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {"classifier": clf, "label_encoder": le}
    with open(_CLASSIFIER_PATH, "wb") as f:
        pickle.dump(payload, f)

    console.print(Panel(
        f"Saved to [bold]{_CLASSIFIER_PATH}[/bold]\n\n"
        f"CV accuracy: [bold green]{cv_scores.mean():.1%}[/bold green]\n\n"
        f"The pipeline will now automatically predict the incoming ticket's category\n"
        f"and filter retrieval to that bucket — expect Category Precision@1 to jump\n"
        f"from ~64% toward 90%+.",
        title="[bold green]Classifier trained[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
