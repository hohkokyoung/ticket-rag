"""
Cross-encoder reranker using BAAI/bge-reranker-base.
Supports multilingual content.

Cross-encoders jointly encode (query, passage) pairs — much more accurate
than bi-encoder cosine similarity alone, at the cost of O(n) inference calls.
We apply this only to the top-k from hybrid search to keep latency acceptable.
"""

from __future__ import annotations

import os
import numpy as np


_DEFAULT_RERANKER = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
_TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))


class Reranker:
    def __init__(self, model_name: str = _DEFAULT_RERANKER):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)

    def rerank(self, query: str, hits: list[dict], top_k: int = _TOP_K_RERANK) -> list[dict]:
        """
        Rerank hits using cross-encoder scores.
        Returns top_k hits sorted by cross-encoder score descending,
        with 'rerank_score' added to each dict.
        """
        if not hits:
            return []

        self._load()

        pairs = [(query, hit["text"]) for hit in hits]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)

        reranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        return reranked[:top_k]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
