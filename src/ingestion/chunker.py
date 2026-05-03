"""
Semantic chunking: splits long ticket text at points where cosine similarity
between adjacent sentence embeddings drops significantly (semantic boundary).
Short tickets are kept as a single chunk.
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_CHUNK_CHARS = 100
_MAX_CHUNK_CHARS = 1500
_SIMILARITY_THRESHOLD = 0.75  # drop below this → new chunk


@dataclass
class Chunk:
    text: str
    ticket_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return f"{self.ticket_id}__chunk{self.chunk_index}"


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(np.dot(a, b) / denom)


def _merge_sentences(sentences: list[str], embeddings: np.ndarray) -> list[str]:
    """Group sentences into chunks based on semantic similarity drops."""
    if len(sentences) <= 1:
        return sentences

    chunks: list[list[str]] = [[sentences[0]]]

    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        current_chunk_text = " ".join(chunks[-1])

        too_long = len(current_chunk_text) + len(sentences[i]) > _MAX_CHUNK_CHARS
        semantic_break = sim < _SIMILARITY_THRESHOLD

        if (semantic_break or too_long) and len(current_chunk_text) >= _MIN_CHUNK_CHARS:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    return [" ".join(group) for group in chunks]


def chunk_ticket(ticket: dict, embed_fn) -> list[Chunk]:
    """
    Chunk a single ticket dict. embed_fn(list[str]) -> np.ndarray of shape (n, dim).
    Returns a list of Chunk objects.
    """
    text = ticket["text"]
    if not text.strip():
        return []

    # Short tickets: single chunk
    if len(text) <= _MAX_CHUNK_CHARS:
        return [Chunk(
            text=text,
            ticket_id=ticket["ticket_id"],
            chunk_index=0,
            metadata={
                "category": ticket.get("category", ""),
                "language": ticket.get("language", ""),
                "priority": ticket.get("priority", ""),
                "subject": ticket.get("subject", ""),
                "has_resolution": bool(ticket.get("resolution", "")),
            },
        )]

    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return [Chunk(
            text=text,
            ticket_id=ticket["ticket_id"],
            chunk_index=0,
            metadata={
                "category": ticket.get("category", ""),
                "language": ticket.get("language", ""),
                "priority": ticket.get("priority", ""),
                "subject": ticket.get("subject", ""),
                "has_resolution": bool(ticket.get("resolution", "")),
            },
        )]

    # Embed sentences to find semantic boundaries
    embeddings = embed_fn([f"passage: {s}" for s in sentences])
    merged = _merge_sentences(sentences, embeddings)

    chunks = []
    for i, chunk_text in enumerate(merged):
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text,
                ticket_id=ticket["ticket_id"],
                chunk_index=i,
                metadata={
                    "category": ticket.get("category", ""),
                    "language": ticket.get("language", ""),
                    "priority": ticket.get("priority", ""),
                    "subject": ticket.get("subject", ""),
                    "has_resolution": bool(ticket.get("resolution", "")),
                },
            ))

    return chunks


def chunk_tickets(tickets: list[dict], embed_fn, show_progress: bool = True) -> list[Chunk]:
    """Chunk all tickets. Uses embed_fn in batches for efficiency."""
    all_chunks: list[Chunk] = []

    iterator = tickets
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(tickets, desc="Semantic chunking")

    for ticket in iterator:
        all_chunks.extend(chunk_ticket(ticket, embed_fn))

    return all_chunks
