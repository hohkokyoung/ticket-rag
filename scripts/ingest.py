"""
One-shot ingestion: load tickets → semantic chunk → embed → index in ChromaDB + BM25.

Resume support: if interrupted mid-embedding, re-running continues from the last
completed batch instead of starting over. Delete .ingest_checkpoint.pkl to force
a full re-ingest.
"""

import sys
import os
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from tqdm import tqdm

from src.ingestion.loader import load_tickets
from src.ingestion.chunker import chunk_tickets, Chunk
from src.ingestion.embedder import get_embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index


_BATCH_SIZE = 256
_CHECKPOINT_PATH = Path(".ingest_checkpoint.pkl")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "text": c.text,
            "ticket_id": c.ticket_id,
            "chunk_index": c.chunk_index,
            "metadata": c.metadata,
        }
        for c in chunks
    ]


def _dicts_to_chunks(dicts: list[dict]) -> list[Chunk]:
    return [
        Chunk(
            text=d["text"],
            ticket_id=d["ticket_id"],
            chunk_index=d["chunk_index"],
            metadata=d["metadata"],
        )
        for d in dicts
    ]


def _save_checkpoint(chunks: list[Chunk], done_batches: set[int]) -> None:
    with open(_CHECKPOINT_PATH, "wb") as f:
        pickle.dump(
            {"chunks": _chunks_to_dicts(chunks), "done_batches": done_batches},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _load_checkpoint() -> tuple[list[Chunk] | None, set[int]]:
    if not _CHECKPOINT_PATH.exists():
        return None, set()
    with open(_CHECKPOINT_PATH, "rb") as f:
        data = pickle.load(f)
    chunks = _dicts_to_chunks(data["chunks"])
    done_batches = data["done_batches"]
    print(f"Resuming from checkpoint: {len(chunks)} chunks, {len(done_batches)} batches already embedded.")
    return chunks, done_batches


def _delete_checkpoint() -> None:
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()


# ── Main ──────────────────────────────────────────────────────────────────────

def _get_data_dir() -> str:
    """Use data/train/ if a holdout split exists, otherwise data/."""
    meta_path = Path(".holdout_meta.json")
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        train_dir = meta["train_dir"]
        print(f"Holdout split detected — ingesting train set only: {train_dir}")
        return train_dir
    return "data"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-ingest from scratch, ignoring existing indexes and checkpoint")
    args = parser.parse_args()

    vs = VectorStore()
    bm25 = BM25Index()

    # Already fully complete?
    if not args.force and not vs.is_empty() and bm25.load():
        print(f"Indexes already populated ({vs.count} vectors). Use --force to re-ingest.")
        return

    if args.force:
        print("--force: clearing checkpoint and existing indexes.")
        _delete_checkpoint()

    # ── Step 1: Chunks (reuse from checkpoint if available) ──────────────────
    chunks, done_batches = _load_checkpoint()

    if chunks is None:
        data_dir = _get_data_dir()
        print(f"Loading tickets from {data_dir}/...")
        tickets = load_tickets(data_dir)
        print(f"Loaded {len(tickets)} tickets")

        embedder = get_embedder()
        print(f"Embedding model: {embedder.model_name}")

        print("Semantic chunking...")
        chunks = chunk_tickets(tickets, embedder.embed_fn, show_progress=True)
        print(f"Generated {len(chunks)} chunks from {len(tickets)} tickets")

        # Save chunks to checkpoint immediately — the expensive part next is embedding
        _save_checkpoint(chunks, done_batches=set())
        print(f"Checkpoint saved to {_CHECKPOINT_PATH}")
    else:
        embedder = get_embedder()

    # ── Step 2: Embed + upsert in batches (skips completed batches) ──────────
    texts = [c.text for c in chunks]
    batch_starts = list(range(0, len(texts), _BATCH_SIZE))
    remaining = [s for s in batch_starts if s not in done_batches]

    if remaining:
        print(f"Embedding {len(remaining)}/{len(batch_starts)} batches "
              f"({len(done_batches)} already done)...")

        for start in tqdm(remaining, desc="Embedding + indexing batches"):
            batch_chunks = chunks[start : start + _BATCH_SIZE]
            batch_texts  = texts[start : start + _BATCH_SIZE]

            embs = embedder.embed_passages(batch_texts)
            vs.upsert_chunks(batch_chunks, embs)

            done_batches.add(start)
            _save_checkpoint(chunks, done_batches)   # update after every batch

        print(f"ChromaDB now has {vs.count} vectors")
    else:
        print(f"All {len(batch_starts)} embedding batches already done. Skipping to BM25.")

    # ── Step 3: BM25 (always rebuilt from chunks so it stays in sync) ────────
    print("Building BM25 index...")
    doc_ids = [c.doc_id for c in chunks]
    bm25.build(doc_ids, texts)
    bm25.save()
    print("BM25 index saved to .bm25_index.pkl")

    # ── Done — clean up checkpoint ────────────────────────────────────────────
    _delete_checkpoint()
    print(f"\nIngestion complete ({vs.count} vectors). Run `python main.py` to query.")


if __name__ == "__main__":
    main()
