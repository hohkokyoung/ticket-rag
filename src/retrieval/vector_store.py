"""ChromaDB vector store: upsert and query ticket chunks."""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.ingestion.chunker import Chunk


_COLLECTION_NAME = "tickets"
_CHROMA_PATH = os.getenv("CHROMA_PATH", ".chroma")
_TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "20"))


class VectorStore:
    def __init__(self, path: str = _CHROMA_PATH):
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def is_empty(self) -> bool:
        return self.count == 0

    def upsert_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Batch upsert chunks and their precomputed embeddings."""
        if not chunks:
            return

        ids = [c.doc_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "ticket_id": c.ticket_id,
                "chunk_index": c.chunk_index,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]
        emb_list = embeddings.tolist()

        # Chroma has a batch size limit; upsert in pages of 5000
        page = 5000
        for start in range(0, len(ids), page):
            self._collection.upsert(
                ids=ids[start : start + page],
                documents=texts[start : start + page],
                embeddings=emb_list[start : start + page],
                metadatas=metadatas[start : start + page],
            )

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = _TOP_K_DENSE,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Return top-n results as list of dicts:
        {doc_id, text, ticket_id, chunk_index, score, metadata}
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(n_results, self.count or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        hits = []
        for doc_id, text, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            # Chroma cosine distance → similarity score
            score = 1.0 - dist
            hits.append({
                "doc_id": doc_id,
                "text": text,
                "ticket_id": meta.get("ticket_id", ""),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": score,
                "metadata": meta,
            })

        return hits
