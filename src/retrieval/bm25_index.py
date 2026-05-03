"""
BM25 sparse index over ticket chunks.
Persists to disk as a pickle so it survives restarts without re-indexing.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


_INDEX_PATH = os.getenv("BM25_INDEX_PATH", ".bm25_index.pkl")
_TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "20"))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._texts: list[str] = []

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        self._doc_ids = doc_ids
        self._texts = texts
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def query(self, query: str, n_results: int = _TOP_K_SPARSE) -> list[dict]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Run build() first.")

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:n_results]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                hits.append({
                    "doc_id": self._doc_ids[idx],
                    "text": self._texts[idx],
                    "score": float(scores[idx]),
                })

        return hits

    def save(self, path: str = _INDEX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "doc_ids": self._doc_ids, "texts": self._texts},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path: str = _INDEX_PATH) -> bool:
        """Load from disk. Returns True if successful."""
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._doc_ids = data["doc_ids"]
        self._texts = data["texts"]
        return True

    def is_empty(self) -> bool:
        return self._bm25 is None
