"""
RAG Evaluation Suite
====================
A. Retrieval metrics   — Recall@K, MRR
B. Answer quality      — Semantic similarity, LLM judge (1-5)
C. RAGAS-style metrics — Faithfulness, Answer relevance

Usage:
    python scripts/eval.py                          # default: 50 samples, all metrics
    python scripts/eval.py --n-samples 20           # faster run
    python scripts/eval.py --skip-generation        # retrieval metrics only (fastest)
    python scripts/eval.py --skip-llm-judge         # skip the slow LLM scoring
    python scripts/eval.py --k 1 3 5 10             # custom Recall@K values
"""

import sys
import os
import argparse
import random
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def build_eval_samples(n: int, seed: int = 42) -> list[dict]:
    """
    Sample N tickets that have resolutions.
    Each sample: { ticket_id, query (subject+body), ground_truth (resolution), category, language }
    """
    from src.ingestion.loader import load_tickets
    tickets = load_tickets("data")
    with_resolution = [t for t in tickets if t.get("resolution", "").strip()]
    random.seed(seed)
    sampled = random.sample(with_resolution, min(n, len(with_resolution)))
    return [
        {
            "ticket_id": t["ticket_id"],
            "query":     f"{t['subject']}\n{t['body']}".strip(),
            "ground_truth": t["resolution"],
            "category":  t.get("category", ""),
            "language":  t.get("language", ""),
        }
        for t in sampled
    ]


def build_holdout_samples(n: int, seed: int = 42) -> list[dict]:
    """
    Load genuinely unseen test tickets from the holdout split.
    These were never ingested — retrieval must generalise to find similar ones.
    """
    meta_path = Path(".holdout_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            "No holdout split found. Run `python scripts/prepare_holdout.py` first."
        )

    import json
    from src.ingestion.loader import load_tickets

    meta = json.loads(meta_path.read_text())
    test_dir = meta["test_dir"]
    tickets = load_tickets(test_dir)
    with_resolution = [t for t in tickets if t.get("resolution", "").strip()]

    random.seed(seed)
    sampled = random.sample(with_resolution, min(n, len(with_resolution)))

    console.print(
        f"[dim]Holdout test set: {len(tickets)} tickets total, "
        f"using {len(sampled)} with resolutions[/dim]"
    )
    return [
        {
            "ticket_id":    t["ticket_id"],
            "query":        f"{t['subject']}\n{t['body']}".strip(),
            "ground_truth": t["resolution"],
            "category":     t.get("category", ""),
            "language":     t.get("language", ""),
        }
        for t in sampled
    ]


# ── A. Retrieval metrics ───────────────────────────────────────────────────────

def eval_retrieval(samples: list[dict], pipeline, k_values: list[int]) -> dict:
    """
    Standard eval (non-holdout):
      Recall@K — did we retrieve the source ticket in top K?
      MRR      — mean reciprocal rank of the source ticket.
    """
    max_k = max(k_values)
    recall_hits = {k: 0 for k in k_values}
    reciprocal_ranks = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        task = progress.add_task("Retrieval eval...", total=len(samples))

        for sample in samples:
            hits = pipeline.retrieve_only(sample["query"], top_k=max_k)
            retrieved_ids = [
                h.get("ticket_id") or h["doc_id"].split("__chunk")[0]
                for h in hits
            ]

            for k in k_values:
                if sample["ticket_id"] in retrieved_ids[:k]:
                    recall_hits[k] += 1

            rank = next(
                (i + 1 for i, tid in enumerate(retrieved_ids) if tid == sample["ticket_id"]),
                None,
            )
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            progress.advance(task)

    n = len(samples)
    return {
        "mode":   "standard",
        "recall": {k: recall_hits[k] / n for k in k_values},
        "mrr":    float(np.mean(reciprocal_ranks)),
        "n":      n,
    }


def eval_retrieval_holdout(
    samples: list[dict],
    pipeline,
    k_values: list[int],
    embedder=None,
    sim_threshold: float = 0.75,   # kept for API compat, unused
) -> dict:
    """
    Holdout eval — Category Precision@K + MRR.

    Ground truth: a retrieved ticket is relevant if it shares the same support
    category as the query ticket.  With 4 categories, random chance = 25%, so
    100%@1 is a genuine strong signal.

    Why not Semantic Recall@K:
      Dense retrieval is literally optimized to return high cosine-similarity
      results, so any cosine threshold becomes circular — you always get ~100%.
      Category Precision is the only non-circular relevance signal available
      without human-labeled judgments.

    Note on MMR: eval does NOT pass use_mmr=True (default is off), so MMR
    diversification does not affect these numbers.
    """
    max_k = max(k_values)
    precision_hits = {k: 0 for k in k_values}
    reciprocal_ranks = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        task = progress.add_task("Retrieval eval (holdout)...", total=len(samples))

        for sample in samples:
            hits = pipeline.retrieve_only(sample["query"], top_k=max_k)
            query_category = sample["category"]

            # A hit is relevant if it shares the same support category
            relevant = [
                h.get("metadata", {}).get("category", "") == query_category
                for h in hits
            ]

            # Category Precision@K — fraction of top-K hits that are relevant
            for k in k_values:
                top_k_rel = relevant[:k]
                precision_hits[k] += sum(top_k_rel) / k if top_k_rel else 0.0

            # MRR — rank of first relevant hit
            rank = next((i + 1 for i, r in enumerate(relevant) if r), None)
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            progress.advance(task)

    n = len(samples)
    return {
        "mode":      "holdout",
        "recall":    {k: precision_hits[k] / n for k in k_values},
        "mrr":       float(np.mean(reciprocal_ranks)),
        "n":         n,
        "threshold": sim_threshold,   # retained in output for schema compat
    }


# ── B. Answer quality ──────────────────────────────────────────────────────────

def eval_semantic_similarity(samples: list[dict], pipeline, embedder) -> dict:
    """
    Embed generated answer + ground truth resolution → cosine similarity.
    Higher = answer is closer in meaning to the known resolution.
    """
    scores = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        task = progress.add_task("Semantic similarity...", total=len(samples))

        for sample in samples:
            result = pipeline.query(sample["query"])
            sample["_generated"] = result.answer  # cache for LLM judge + faithfulness
            sample["_hits"]      = result.hits    # cache for faithfulness (avoids second query)

            emb_gen = embedder.embed_query(result.answer)
            emb_gt  = embedder.embed_query(sample["ground_truth"])
            scores.append(cosine(emb_gen, emb_gt))
            progress.advance(task)

    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "scores": scores,
    }


def eval_llm_judge(samples: list[dict]) -> dict:
    """
    Ask Groq / Claude to rate the generated answer vs ground truth on a 1-5 scale.
    Samples must already have _generated set from semantic similarity eval.
    """
    _JUDGE_PROMPT = """You are an evaluation judge for a customer support RAG system.

Rate the Generated Resolution against the Reference Resolution on a scale of 1 to 5.

Scoring rubric:
5 — Addresses the issue completely, accurate, actionable, matches or improves on reference
4 — Mostly correct with minor gaps or slightly less specific
3 — Partially addresses the issue, key steps present but incomplete
2 — Somewhat relevant but missing critical information
1 — Incorrect, unhelpful, or off-topic

Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}"""

    scores = []
    reasons = []

    def _call_judge(query: str, generated: str, ground_truth: str) -> tuple[int, str]:
        prompt = f"""Customer Query:
{query}

Reference Resolution:
{ground_truth}

Generated Resolution:
{generated}"""

        if os.getenv("GROQ_API_KEY"):
            from groq import Groq
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=100,
                temperature=0,
                messages=[
                    {"role": "system", "content": _JUDGE_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content
        elif os.getenv("GEMINI_API_KEY"):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_JUDGE_PROMPT,
                    max_output_tokens=100,
                ),
            )
            raw = resp.text
        elif os.getenv("OLLAMA_MODEL"):
            import urllib.request as _urllib
            payload = json.dumps({
                "model": os.environ["OLLAMA_MODEL"],
                "stream": False,
                "messages": [
                    {"role": "system", "content": _JUDGE_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            }).encode()
            req = _urllib.Request(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with _urllib.urlopen(req, timeout=120) as r:
                raw = json.loads(r.read())["message"]["content"]
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                system=_JUDGE_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text

        try:
            data = json.loads(raw.strip())
            return int(data["score"]), data.get("reason", "")
        except Exception:
            # fallback: try to parse just the number
            for ch in raw:
                if ch.isdigit() and ch in "12345":
                    return int(ch), raw
            return 3, "parse error"

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        task = progress.add_task("LLM judge scoring...", total=len(samples))

        for sample in samples:
            if "_generated" not in sample:
                progress.advance(task)
                continue
            score, reason = _call_judge(
                sample["query"], sample["_generated"], sample["ground_truth"]
            )
            scores.append(score)
            reasons.append(reason)
            time.sleep(0.3)  # light rate-limit protection
            progress.advance(task)

    return {
        "mean": float(np.mean(scores)) if scores else 0.0,
        "distribution": {i: scores.count(i) for i in range(1, 6)},
        "scores": scores,
        "reasons": reasons,
    }


# ── C. RAGAS-style metrics (custom implementation) ────────────────────────────

def eval_faithfulness(samples: list[dict], pipeline, embedder, threshold: float = 0.70) -> dict:
    """
    Faithfulness: fraction of sentences in the generated answer that are semantically
    supported by the retrieved context chunks.

    A sentence is "supported" if its cosine similarity to any retrieved chunk exceeds threshold.
    Threshold of 0.70 (raised from 0.55) — at 0.55 almost anything passes, giving a
    false 1.000 score. 0.70 requires genuine semantic overlap to count as supported.
    """
    import re
    sentence_re = re.compile(r"(?<=[.!?])\s+")

    faithfulness_scores = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        task = progress.add_task("Faithfulness...", total=len(samples))

        for sample in samples:
            # Reuse cached answer + hits from eval_semantic_similarity if available —
            # avoids a second pipeline.query() call (and second LLM generation).
            if "_generated" in sample and "_hits" in sample:
                generated = sample["_generated"]
                hits = sample["_hits"]
            else:
                result = pipeline.query(sample["query"])
                generated = result.answer
                hits = result.hits
                sample["_generated"] = generated
                sample["_hits"] = hits

            # Split answer into sentences
            sentences = [s.strip() for s in sentence_re.split(generated) if s.strip()]
            if not sentences:
                faithfulness_scores.append(1.0)
                progress.advance(task)
                continue

            # Embed all sentences + all retrieved chunk texts
            context_texts = [h["text"] for h in hits]
            all_texts = sentences + context_texts
            all_embs  = embedder.embed_passages(all_texts)

            sent_embs    = all_embs[:len(sentences)]
            context_embs = all_embs[len(sentences):]

            supported = 0
            for sent_emb in sent_embs:
                sims = [cosine(sent_emb, ctx_emb) for ctx_emb in context_embs]
                if max(sims) >= threshold:
                    supported += 1

            faithfulness_scores.append(supported / len(sentences))
            progress.advance(task)

    return {
        "mean": float(np.mean(faithfulness_scores)),
        "median": float(np.median(faithfulness_scores)),
        "scores": faithfulness_scores,
    }


def eval_answer_relevance(samples: list[dict], pipeline, embedder) -> dict:
    """
    Answer Relevance: cosine similarity between the generated answer and the original query.
    Higher = the answer actually addresses what was asked.
    """
    scores = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        task = progress.add_task("Answer relevance...", total=len(samples))

        for sample in samples:
            if "_generated" in sample:
                generated = sample["_generated"]
            else:
                result = pipeline.query(sample["query"])
                generated = result.answer
                sample["_generated"] = generated

            emb_answer = embedder.embed_query(generated)
            emb_query  = embedder.embed_query(sample["query"])
            scores.append(cosine(emb_answer, emb_query))
            progress.advance(task)

    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "scores": scores,
    }


# ── Display ────────────────────────────────────────────────────────────────────

def _score_color(val: float, low: float = 0.4, high: float = 0.7) -> str:
    if val >= high:
        return "green"
    if val >= low:
        return "yellow"
    return "red"


def print_results(
    retrieval: dict | None,
    similarity: dict | None,
    llm_judge: dict | None,
    faithfulness: dict | None,
    answer_relevance: dict | None,
    k_values: list[int],
):
    console.print()

    # ── A. Retrieval ──────────────────────────────────────────────────────────
    if retrieval:
        is_holdout = retrieval.get("mode") == "holdout"
        title = "A. Retrieval Metrics (Holdout)" if is_holdout else "A. Retrieval Metrics"
        t = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold cyan")
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
        t.add_column("Interpretation", style="dim")

        if is_holdout:
            for k in k_values:
                v = retrieval["recall"][k]
                color = _score_color(v, low=0.5, high=0.8)
                t.add_row(
                    f"Category Precision@{k}",
                    f"[{color}]{v:.1%}[/{color}]",
                    f"Fraction of top-{k} hits sharing the query's support category (random = 25%)",
                )
        else:
            for k in k_values:
                v = retrieval["recall"][k]
                color = _score_color(v, low=0.5, high=0.8)
                t.add_row(
                    f"Recall@{k}",
                    f"[{color}]{v:.1%}[/{color}]",
                    f"{int(v * retrieval['n'])}/{retrieval['n']} queries found the source ticket in top {k}",
                )

        mrr = retrieval["mrr"]
        color = _score_color(mrr, low=0.3, high=0.6)
        mrr_label = "Category MRR" if is_holdout else "MRR"
        mrr_desc  = "Reciprocal rank of first same-category hit" if is_holdout else "Mean reciprocal rank — higher = source ticket ranked earlier"
        t.add_row(mrr_label, f"[{color}]{mrr:.4f}[/{color}]", mrr_desc)
        console.print(t)

    # ── B. Answer Quality ─────────────────────────────────────────────────────
    if similarity or llm_judge:
        t = Table(title="B. Answer Quality", box=box.SIMPLE_HEAVY, header_style="bold cyan")
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
        t.add_column("Interpretation", style="dim")

        if similarity:
            v = similarity["mean"]
            color = _score_color(v, low=0.4, high=0.65)
            t.add_row(
                "Semantic Similarity (mean)",
                f"[{color}]{v:.4f}[/{color}]",
                "Cosine similarity vs ground-truth resolution (0→1)",
            )
            t.add_row(
                "Semantic Similarity (median)",
                f"{similarity['median']:.4f}",
                f"Range: {similarity['min']:.3f} – {similarity['max']:.3f}",
            )

        if llm_judge:
            v = llm_judge["mean"]
            color = _score_color(v / 5, low=0.5, high=0.7)
            dist = llm_judge["distribution"]
            dist_str = "  ".join(f"{k}★:{dist.get(k,0)}" for k in range(1, 6))
            t.add_row(
                "LLM Judge (mean)",
                f"[{color}]{v:.2f} / 5[/{color}]",
                dist_str,
            )

        console.print(t)

    # ── C. RAGAS-style ────────────────────────────────────────────────────────
    if faithfulness or answer_relevance:
        t = Table(title="C. RAGAS-style Metrics", box=box.SIMPLE_HEAVY, header_style="bold cyan")
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
        t.add_column("Interpretation", style="dim")

        if faithfulness:
            v = faithfulness["mean"]
            color = _score_color(v, low=0.5, high=0.75)
            t.add_row(
                "Faithfulness (mean)",
                f"[{color}]{v:.4f}[/{color}]",
                "Fraction of answer sentences supported by retrieved context",
            )

        if answer_relevance:
            v = answer_relevance["mean"]
            color = _score_color(v, low=0.4, high=0.65)
            t.add_row(
                "Answer Relevance (mean)",
                f"[{color}]{v:.4f}[/{color}]",
                "Cosine similarity between answer and original query",
            )

        console.print(t)

    # ── Score legend ──────────────────────────────────────────────────────────
    console.print("[green]■[/green] Good  [yellow]■[/yellow] Needs work  [red]■[/red] Poor\n", style="dim")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the ticket RAG pipeline")
    parser.add_argument("--n-samples",       type=int,   default=50,       help="Number of tickets to sample (default: 50)")
    parser.add_argument("--k",               type=int,   nargs="+",        default=[1, 3, 5, 10], help="K values for Recall@K / Precision@K")
    parser.add_argument("--skip-generation", action="store_true",          help="Skip all generation metrics (retrieval only)")
    parser.add_argument("--skip-llm-judge",  action="store_true",          help="Skip LLM judge scoring")
    parser.add_argument("--holdout",         action="store_true",          help="Evaluate on unseen holdout test set (run prepare_holdout.py first)")
    parser.add_argument("--seed",            type=int,   default=42,       help="Random seed for sample selection")
    parser.add_argument("--sim-threshold",   type=float, default=0.75,     help="Cosine similarity threshold for holdout relevance (default: 0.75). Lower = easier bar; raise to 0.80+ for stricter eval.")
    args = parser.parse_args()

    # Load pipeline
    with console.status("[bold green]Loading pipeline..."):
        from src.pipeline import build_pipeline
        from src.ingestion.embedder import get_embedder
        pipeline = build_pipeline()
        embedder = get_embedder()
    console.print(f"[green]Pipeline ready.[/green] Sampling [bold]{args.n_samples}[/bold] tickets...\n")

    # Build eval set
    if args.holdout:
        console.print("[bold yellow]Mode: Holdout (genuinely unseen test tickets)[/bold yellow]")
        samples = build_holdout_samples(args.n_samples, seed=args.seed)
    else:
        console.print("[bold]Mode: Standard (self-retrieval, inflated baseline)[/bold]")
        samples = build_eval_samples(args.n_samples, seed=args.seed)

    console.print(
        f"Eval set: {len(samples)} tickets  |  "
        f"Languages: {', '.join(sorted(set(s['language'] for s in samples if s['language']))[:6])}  |  "
        f"Categories: {len(set(s['category'] for s in samples))} unique\n"
    )

    retrieval_results    = None
    similarity_results   = None
    judge_results        = None
    faithfulness_results = None
    relevance_results    = None

    # ── A ────────────────────────────────────────────────────────────────────
    console.rule("[bold]A. Retrieval Metrics[/bold]")
    if args.holdout:
        retrieval_results = eval_retrieval_holdout(
            samples, pipeline, args.k,
            embedder=embedder,
            sim_threshold=args.sim_threshold,
        )
    else:
        retrieval_results = eval_retrieval(samples, pipeline, args.k)

    if not args.skip_generation:
        # ── B: Semantic similarity ────────────────────────────────────────────
        console.rule("[bold]B. Answer Quality[/bold]")
        similarity_results = eval_semantic_similarity(samples, pipeline, embedder)

        # ── B: LLM judge ─────────────────────────────────────────────────────
        if not args.skip_llm_judge:
            judge_results = eval_llm_judge(samples)

        # ── C ────────────────────────────────────────────────────────────────
        console.rule("[bold]C. RAGAS-style Metrics[/bold]")
        faithfulness_results = eval_faithfulness(samples, pipeline, embedder)
        relevance_results    = eval_answer_relevance(samples, pipeline, embedder)

    # ── Print summary ─────────────────────────────────────────────────────────
    console.rule("[bold]Results[/bold]")
    print_results(
        retrieval_results,
        similarity_results,
        judge_results,
        faithfulness_results,
        relevance_results,
        args.k,
    )


if __name__ == "__main__":
    main()
