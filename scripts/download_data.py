"""
Download the Kaggle multilingual customer support tickets dataset.
Requires KAGGLE_USERNAME and KAGGLE_KEY in environment (or ~/.kaggle/kaggle.json).
"""

import os
import sys
import zipfile
from pathlib import Path


_DATASET = "tobiasbueck/multilingual-customer-support-tickets"
_DATA_DIR = Path("data")


def main():
    _DATA_DIR.mkdir(exist_ok=True)

    # Allow env-based credentials as alternative to kaggle.json
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        cred_file = kaggle_dir / "kaggle.json"
        if not cred_file.exists():
            cred_file.write_text(f'{{"username":"{username}","key":"{key}"}}')
            cred_file.chmod(0o600)
            print(f"Wrote credentials to {cred_file}")

    try:
        import kaggle
    except ImportError:
        print("kaggle package not found. Run: pip install kaggle")
        sys.exit(1)

    print(f"Downloading dataset: {_DATASET}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        _DATASET,
        path=str(_DATA_DIR),
        unzip=True,
        quiet=False,
    )

    csvs = list(_DATA_DIR.glob("**/*.csv"))
    if csvs:
        print(f"\nDownload complete. Found {len(csvs)} CSV file(s):")
        for csv in csvs:
            size_mb = csv.stat().st_size / 1_048_576
            print(f"  {csv}  ({size_mb:.1f} MB)")
    else:
        print("\nWarning: no CSV files found after download. Check the data/ directory.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
