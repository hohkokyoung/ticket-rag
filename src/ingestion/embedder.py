"""
Embedding wrapper around sentence-transformers multilingual-e5-large.

multilingual-e5 requires query prefix "query: " and passage prefix "passage: "
for optimal retrieval quality. This module handles that automatically.
"""

from __future__ import annotations

import os
import numpy as np
from functools import cached_property


_DEFAULT_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
_BATCH_SIZE = 64


class Embedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed document/passage texts. Adds 'passage: ' prefix."""
        self._load()
        prefixed = [f"passage: {t}" if not t.startswith("passage: ") else t for t in texts]
        return self._model.encode(
            prefixed,
            batch_size=_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string. Adds 'query: ' prefix."""
        self._load()
        prefixed = f"query: {text}" if not text.startswith("query: ") else text
        return self._model.encode(
            [prefixed],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

    def embed_fn(self, texts: list[str]) -> np.ndarray:
        """Generic embed function for the chunker (expects already-prefixed texts)."""
        self._load()
        return self._model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    @cached_property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()


# Module-level singleton
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
