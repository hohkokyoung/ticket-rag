"""Loads and normalizes the Kaggle multilingual customer support ticket dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pandas as pd


# Columns we care about — gracefully handle missing ones
_PREFERRED_COLS = {
    "ticket_id": ["ticket_id", "id"],
    "subject": ["subject", "title", "summary"],
    "body": ["body", "description", "text", "content", "message"],
    "resolution": ["resolution", "response", "answer", "solution", "reply"],
    "category": ["category", "type", "issue_type", "topic"],
    "language": ["language", "lang"],
    "priority": ["priority", "severity"],
}


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names and fill gaps."""
    rename_map: dict[str, str] = {}
    for canonical, candidates in _PREFERRED_COLS.items():
        found = _pick_col(df, candidates)
        if found and found != canonical:
            rename_map[found] = canonical

    df = df.rename(columns=rename_map)

    # Ensure all canonical columns exist
    for col in _PREFERRED_COLS:
        if col not in df.columns:
            df[col] = ""

    # Generate ticket_id if missing / all-null
    if df["ticket_id"].isna().all() or (df["ticket_id"] == "").all():
        df["ticket_id"] = [f"ticket_{i}" for i in range(len(df))]
    else:
        df["ticket_id"] = df["ticket_id"].astype(str)

    df = df.fillna("")
    return df


def _build_text(row: pd.Series) -> str:
    """Combine ticket fields into a single rich text block for embedding."""
    parts = []
    if row["subject"]:
        parts.append(f"Subject: {row['subject']}")
    if row["body"]:
        parts.append(f"Problem: {row['body']}")
    if row["resolution"]:
        parts.append(f"Resolution: {row['resolution']}")
    return "\n\n".join(parts)


def load_tickets(data_dir: str | Path = "data") -> list[dict]:
    """
    Load all CSVs in data_dir, normalize columns, and return a list of ticket dicts.
    Each dict has: ticket_id, subject, body, resolution, category, language, priority, text
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Run `python scripts/download_data.py` first."
        )

    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = _normalize(df)

    # Drop rows with no usable content
    df = df[df.apply(lambda r: bool(r["subject"] or r["body"]), axis=1)]
    df = df.drop_duplicates(subset=["ticket_id"])

    tickets = []
    for _, row in df.iterrows():
        tickets.append({
            "ticket_id": row["ticket_id"],
            "subject": row["subject"],
            "body": row["body"],
            "resolution": row["resolution"],
            "category": row["category"],
            "language": row["language"],
            "priority": row["priority"],
            "text": _build_text(row),
        })

    return tickets
