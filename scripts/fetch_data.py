from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "data"
KAGGLE_CREDENTIALS = Path.home() / ".kaggle" / "kaggle.json"
DATASET_SLUG = "adityajn105/flickr8k"


def fetch_flickr8k_data() -> Path:
    if not KAGGLE_CREDENTIALS.exists():
        print(
            f"error: Kaggle API credentials not found at {KAGGLE_CREDENTIALS}.\n"
            "Set up your Kaggle API token: https://www.kaggle.com/docs/api#authentication",
            file=sys.stderr,
        )
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    return DATA_DIR


if __name__ == "__main__":
    fetch_flickr8k_data()
