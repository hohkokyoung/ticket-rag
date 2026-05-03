"""Main RAG pipeline: retrieve → rerank → generate."""

from __future__ import annotations

import os
from dataclasses import dataclass


_TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))


@dataclass
class RAGResult:
    query: str
    hits: list[dict]
    answer: str


class RAGPipeline:
    def __init__(
        self,
        vector_store,
        bm25_index,
        embedder,
        reranker,
    ):
        self._vs = vector_store
        self._bm25 = bm25_index
        self._embedder = embedder
        self._reranker = reranker

    def query(
        self,
        ticket_text: str,
        top_k: int = _TOP_K_RERANK,
        language_filter: str | None = None,
        category_filter: str | None = None,
    ) -> RAGResult:
        from src.retrieval.hybrid_search import hybrid_search
        from src.generation.claude_client import generate_response

        # Optional metadata filters for ChromaDB
        where: dict | None = None
        filters = {}
        if language_filter:
            filters["language"] = language_filter
        if category_filter:
            filters["category"] = category_filter
        if filters:
            if len(filters) == 1:
                where = filters
            else:
                where = {"$and": [{k: v} for k, v in filters.items()]}

        # 1. Embed query
        query_embedding = self._embedder.embed_query(ticket_text)

        # 2. Hybrid search (dense + BM25 → RRF)
        fused_hits = hybrid_search(
            query=ticket_text,
            query_embedding=query_embedding,
            vector_store=self._vs,
            bm25_index=self._bm25,
            where=where,
        )

        # 3. Cross-encoder rerank
        reranked = self._reranker.rerank(ticket_text, fused_hits, top_k=top_k)

        # 4. Generate answer with Claude
        answer = generate_response(ticket_text, reranked)

        return RAGResult(query=ticket_text, hits=reranked, answer=answer)

    def retrieve_only(
        self,
        ticket_text: str,
        top_k: int = _TOP_K_RERANK,
    ) -> list[dict]:
        """Retrieve and rerank without calling Claude (useful for testing retrieval quality)."""
        from src.retrieval.hybrid_search import hybrid_search

        query_embedding = self._embedder.embed_query(ticket_text)
        fused_hits = hybrid_search(
            query=ticket_text,
            query_embedding=query_embedding,
            vector_store=self._vs,
            bm25_index=self._bm25,
        )
        return self._reranker.rerank(ticket_text, fused_hits, top_k=top_k)


def build_pipeline() -> RAGPipeline:
    """Construct a fully-loaded RAGPipeline from persisted indexes."""
    from src.ingestion.embedder import get_embedder
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_index import BM25Index
    from src.retrieval.reranker import get_reranker

    embedder = get_embedder()
    vs = VectorStore()
    bm25 = BM25Index()

    if vs.is_empty():
        raise RuntimeError(
            "Vector store is empty. Run `python scripts/ingest.py` first."
        )

    if not bm25.load():
        raise RuntimeError(
            "BM25 index not found. Run `python scripts/ingest.py` first."
        )

    reranker = get_reranker()
    return RAGPipeline(vs, bm25, embedder, reranker)
