"""
Reciprocal Rank Fusion (RRF) to merge dense and sparse search results.

RRF score = sum(1 / (k + rank_i)) across all result lists.
k=60 is the standard constant from the original RRF paper.
"""

from __future__ import annotations


_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = _RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked result lists via RRF.

    Each result dict must have a "doc_id" key.
    Returns merged list sorted by RRF score descending,
    with the original result dict from the highest-scoring list merged in.
    """
    rrf_scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            doc_id = hit["doc_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            # Keep whichever version of the doc we've seen, prefer richer ones
            if doc_id not in doc_data or len(hit.get("text", "")) > len(doc_data[doc_id].get("text", "")):
                doc_data[doc_id] = hit

    merged = []
    for doc_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = {**doc_data[doc_id], "rrf_score": rrf_score}
        merged.append(entry)

    return merged


def hybrid_search(
    query: str,
    query_embedding,
    vector_store,
    bm25_index,
    n_dense: int | None = None,
    n_sparse: int | None = None,
    where: dict | None = None,
) -> list[dict]:
    """
    Run dense + sparse search and fuse results with RRF.
    Returns fused list sorted by RRF score.
    """
    import os

    n_dense = n_dense or int(os.getenv("TOP_K_DENSE", "20"))
    n_sparse = n_sparse or int(os.getenv("TOP_K_SPARSE", "20"))

    dense_hits = vector_store.query(query_embedding, n_results=n_dense, where=where)
    sparse_hits = bm25_index.query(query, n_results=n_sparse)

    return reciprocal_rank_fusion([dense_hits, sparse_hits])
