"""
Split the dataset into train (80%) and test (20%) sets.
Stratified by category so every category stays represented in both splits.

After running this:
  1. python scripts/ingest.py --force   (re-ingests train only)
  2. python scripts/eval.py --holdout   (evaluates on unseen test tickets)

Saves:
  data/train/   → tickets to ingest
  data/test/    → held-out tickets for eval (never indexed)
  .holdout_meta.json → split metadata
"""

import sys, os, json, random, shutil
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.ingestion.loader import load_tickets

_TEST_RATIO  = 0.20
_SEED        = 42
_META_PATH   = Path(".holdout_meta.json")
_TRAIN_DIR   = Path("data/train")
_TEST_DIR    = Path("data/test")


def main():
    if _META_PATH.exists():
        print(f"Holdout split already exists ({_META_PATH}). Delete it to re-split.")
        meta = json.loads(_META_PATH.read_text())
        print(f"  Train: {meta['n_train']}  Test: {meta['n_test']}")
        return

    print("Loading tickets...")
    tickets = load_tickets("data")
    print(f"Loaded {len(tickets)} total tickets")

    # Stratified split by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    for t in tickets:
        cat = t.get("category", "unknown") or "unknown"
        by_category[cat].append(t)

    random.seed(_SEED)
    train_tickets, test_tickets = [], []

    for cat, cat_tickets in by_category.items():
        random.shuffle(cat_tickets)
        n_test = max(1, int(len(cat_tickets) * _TEST_RATIO))
        test_tickets.extend(cat_tickets[:n_test])
        train_tickets.extend(cat_tickets[n_test:])

    print(f"Split → Train: {len(train_tickets)}  Test: {len(test_tickets)}")

    # Check category distribution
    print("\nCategory distribution:")
    all_cats = sorted(by_category.keys())
    print(f"  {'Category':<30} {'Total':>6} {'Train':>6} {'Test':>5}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*5}")

    test_ids = {t["ticket_id"] for t in test_tickets}
    for cat in all_cats:
        total = len(by_category[cat])
        test  = sum(1 for t in by_category[cat] if t["ticket_id"] in test_ids)
        train = total - test
        print(f"  {cat:<30} {total:>6} {train:>6} {test:>5}")

    # Save splits as CSVs
    _TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    _TEST_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_tickets).to_csv(_TRAIN_DIR / "train.csv", index=False)
    pd.DataFrame(test_tickets).to_csv(_TEST_DIR  / "test.csv",  index=False)

    # Save metadata
    meta = {
        "n_train":    len(train_tickets),
        "n_test":     len(test_tickets),
        "test_ratio": _TEST_RATIO,
        "seed":       _SEED,
        "train_dir":  str(_TRAIN_DIR),
        "test_dir":   str(_TEST_DIR),
        "test_ids":   [t["ticket_id"] for t in test_tickets],
    }
    _META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved:")
    print(f"  {_TRAIN_DIR}/train.csv  ({len(train_tickets)} tickets)")
    print(f"  {_TEST_DIR}/test.csv    ({len(test_tickets)} tickets)")
    print(f"  {_META_PATH}")
    print(f"\nNext steps:")
    print(f"  1. python scripts/ingest.py --force   ← re-ingest train only")
    print(f"  2. python scripts/eval.py --holdout   ← eval on unseen test tickets")


if __name__ == "__main__":
    main()
