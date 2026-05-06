"""
Main RAG pipeline: retrieve → rerank → [MMR] → generate

Improvements applied:
  - TOP_K_RERANK default lowered to 3 (eval showed precision drops sharply after @3)
  - HyDE: optional, embeds a hypothetical resolution instead of raw query for better dense recall
  - MMR: optional, diversifies the reranked results before sending to Claude
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


_TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))
_USE_HYDE     = os.getenv("USE_HYDE", "false").lower() == "true"
_USE_MMR      = os.getenv("USE_MMR", "false").lower() == "true"


@dataclass
class RAGResult:
    query: str
    hits: list[dict]
    answer: str
    hypothetical: str | None = None   # HyDE-generated text, for debugging


class RAGPipeline:
    def __init__(self, vector_store, bm25_index, embedder, reranker, classifier=None):
        self._vs         = vector_store
        self._bm25       = bm25_index
        self._embedder   = embedder
        self._reranker   = reranker
        self._classifier = classifier  # optional — None = unfiltered search

    def query(
        self,
        ticket_text: str,
        top_k: int = _TOP_K_RERANK,
        language_filter: str | None = None,
        category_filter: str | None = None,
        use_hyde: bool = _USE_HYDE,
        use_mmr:  bool = _USE_MMR,
    ) -> RAGResult:
        from src.retrieval.hybrid_search import hybrid_search
        from src.generation.claude_client import generate_response

        # ── Metadata filters ──────────────────────────────────────────────────
        where: dict | None = None
        filters = {}
        if language_filter:
            filters["language"] = language_filter
        if category_filter:
            filters["category"] = category_filter
        if filters:
            where = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]}

        # ── 0. Embed query (needed for classifier + dense search) ─────────────
        # We embed early so the classifier can reuse the same vector.
        hypothetical: str | None = None
        if use_hyde:
            from src.retrieval.hyde import hyde_query_embedding
            dense_embedding, hypothetical = hyde_query_embedding(ticket_text, self._embedder)
        else:
            dense_embedding = self._embedder.embed_query(ticket_text)

        # ── Auto category filter via classifier ───────────────────────────────
        # Only apply if confidence >= threshold (default 0.70).
        # Incident vs Problem are inherently ambiguous — at low confidence the
        # classifier picks the wrong bucket and wipes out all results.
        # Skipping the filter for uncertain predictions is better than a wrong filter.
        _CONFIDENCE_THRESHOLD = 0.70
        if category_filter is None and self._classifier is not None:
            proba = self._classifier.predict_proba(dense_embedding)
            top_prob = max(proba.values())
            if top_prob >= _CONFIDENCE_THRESHOLD:
                predicted = max(proba, key=proba.get)
                if where is None:
                    where = {"category": predicted}
                else:
                    where = {"$and": [where, {"category": predicted}]}

        # ── 1. Hybrid search (dense + BM25 → RRF) ────────────────────────────
        fused_hits = hybrid_search(
            query=ticket_text,               # BM25 always uses original
            query_embedding=dense_embedding,  # dense uses HyDE embedding if enabled
            vector_store=self._vs,
            bm25_index=self._bm25,
            where=where,
        )

        # ── 3. Cross-encoder rerank ───────────────────────────────────────────
        # Rerank against original query (not hypothetical) — we want results
        # that match the actual problem, not just the hypothetical resolution.
        reranked = self._reranker.rerank(ticket_text, fused_hits, top_k=top_k * 3)

        # ── 4. MMR diversification ────────────────────────────────────────────
        # Eval showed top results are often near-duplicates.
        # MMR re-selects top_k from the reranked pool to maximise both
        # relevance and diversity — Claude gets different angles, not 3 copies.
        if use_mmr and len(reranked) > top_k:
            from src.retrieval.mmr import mmr_diversify
            final_hits = mmr_diversify(reranked, self._embedder, top_k=top_k)
        else:
            final_hits = reranked[:top_k]

        # ── 5. Generate ───────────────────────────────────────────────────────
        answer = generate_response(ticket_text, final_hits)

        return RAGResult(
            query=ticket_text,
            hits=final_hits,
            answer=answer,
            hypothetical=hypothetical,
        )

    def retrieve_only(
        self,
        ticket_text: str,
        top_k: int = _TOP_K_RERANK,
        use_hyde: bool = _USE_HYDE,
        use_mmr:  bool = _USE_MMR,
    ) -> list[dict]:
        """Retrieve without calling Claude — for eval and debugging."""
        from src.retrieval.hybrid_search import hybrid_search

        if use_hyde:
            from src.retrieval.hyde import hyde_query_embedding
            dense_embedding, _ = hyde_query_embedding(ticket_text, self._embedder)
        else:
            dense_embedding = self._embedder.embed_query(ticket_text)

        # Auto category filter — only when classifier is confident
        _CONFIDENCE_THRESHOLD = 0.70
        where: dict | None = None
        if self._classifier is not None:
            proba = self._classifier.predict_proba(dense_embedding)
            top_prob = max(proba.values())
            if top_prob >= _CONFIDENCE_THRESHOLD:
                predicted = max(proba, key=proba.get)
                where = {"category": predicted}

        fused_hits = hybrid_search(
            query=ticket_text,
            query_embedding=dense_embedding,
            vector_store=self._vs,
            bm25_index=self._bm25,
            where=where,
        )
        reranked = self._reranker.rerank(ticket_text, fused_hits, top_k=top_k * 3)

        if use_mmr and len(reranked) > top_k:
            from src.retrieval.mmr import mmr_diversify
            return mmr_diversify(reranked, self._embedder, top_k=top_k)

        return reranked[:top_k]


def build_pipeline() -> RAGPipeline:
    from src.ingestion.embedder import get_embedder
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_index import BM25Index
    from src.retrieval.reranker import get_reranker
    from src.retrieval.classifier import load_classifier

    embedder = get_embedder()
    vs   = VectorStore()
    bm25 = BM25Index()

    if vs.is_empty():
        raise RuntimeError("Vector store is empty. Run `python scripts/ingest.py` first.")
    if not bm25.load():
        raise RuntimeError("BM25 index not found. Run `python scripts/ingest.py` first.")

    classifier = load_classifier()
    if classifier:
        from rich.console import Console
        Console().print("[green]Category classifier loaded — retrieval will be filtered by predicted category.[/green]")
    else:
        from rich.console import Console
        Console().print("[dim]No category classifier found. Run `python scripts/train_classifier.py` to enable category filtering.[/dim]")

    return RAGPipeline(vs, bm25, embedder, get_reranker(), classifier=classifier)
