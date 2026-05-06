"""
MMR — Maximal Marginal Relevance

Problem it solves:
  After reranking, the top-K results are often near-duplicates:
  5 tickets about the same thing, phrased slightly differently.
  Claude gets redundant context instead of diverse angles.

How it works:
  Iteratively selects the next result that maximises:
    MMR score = λ * relevance - (1 - λ) * max_similarity_to_already_selected

  λ = 1.0 → pure relevance (same as no MMR)
  λ = 0.0 → pure diversity
  λ = 0.7 → mostly relevance, penalise redundancy  ← default

Result: instead of 3 nearly identical billing tickets, you get
  1 billing ticket + 1 with a different resolution approach + 1 edge case.
  Claude gets broader context → better, more complete answers.

Applied after reranking — reranker scores are used as the relevance signal.
"""

from __future__ import annotations

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _normalize_scores(hits: list[dict]) -> list[float]:
    """Normalize rerank_scores to [0, 1] range for MMR relevance signal."""
    scores = [h.get("rerank_score", h.get("rrf_score", 0.0)) for h in hits]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def mmr_diversify(
    hits: list[dict],
    embedder,
    top_k: int,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Re-select top_k results from hits using MMR to balance relevance and diversity.

    hits     — reranked candidates (must have rerank_score or rrf_score)
    embedder — used to embed hit texts for inter-hit similarity
    top_k    — number of results to return
    lambda   — 0.7 = 70% relevance, 30% diversity penalty
    """
    if len(hits) <= top_k:
        return hits

    # Embed all candidate texts
    texts = [h["text"] for h in hits]
    embeddings = embedder.embed_passages(texts)

    relevance_scores = _normalize_scores(hits)

    selected_indices: list[int] = []
    remaining_indices: list[int] = list(range(len(hits)))

    while len(selected_indices) < top_k and remaining_indices:
        best_idx = None
        best_score = -float("inf")

        for idx in remaining_indices:
            relevance = relevance_scores[idx]

            if not selected_indices:
                # First selection — pure relevance
                mmr_score = relevance
            else:
                # Penalise similarity to already-selected results
                sims = [
                    _cosine(embeddings[idx], embeddings[sel_idx])
                    for sel_idx in selected_indices
                ]
                max_sim = max(sims)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [hits[i] for i in selected_indices]
