"""
Category classifier — predicts the support category of an incoming ticket
from its query embedding, so the pipeline can filter retrieval to that bucket.

Loaded once at pipeline startup; predict() is called per query (microseconds).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

_CLASSIFIER_PATH = Path(".category_classifier.pkl")


class CategoryClassifier:
    def __init__(self, clf, label_encoder):
        self._clf = clf
        self._le  = label_encoder

    def predict(self, embedding: list[float] | np.ndarray) -> str:
        """Return the most likely category for the given query embedding."""
        x = np.array(embedding).reshape(1, -1)
        idx = self._clf.predict(x)[0]
        return self._le.inverse_transform([idx])[0]

    def predict_proba(self, embedding: list[float] | np.ndarray) -> dict[str, float]:
        """Return {category: probability} dict — useful for debugging."""
        x = np.array(embedding).reshape(1, -1)
        probs = self._clf.predict_proba(x)[0]
        return dict(zip(self._le.classes_, probs))


def load_classifier() -> CategoryClassifier | None:
    """
    Load the trained classifier from disk.
    Returns None if not found — pipeline falls back to unfiltered search.
    """
    if not _CLASSIFIER_PATH.exists():
        return None
    with open(_CLASSIFIER_PATH, "rb") as f:
        payload = pickle.load(f)
    return CategoryClassifier(payload["classifier"], payload["label_encoder"])
