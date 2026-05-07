import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment.config import GLOVE_TXT, IMDB_CSV, RAW_DIR  # noqa: E402

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_MEMBER = "glove.6B.100d.txt"


def download_imdb():
    if IMDB_CSV.exists():
        print(f"[skip] IMDB already at {IMDB_CSV}")
        return
    print("Downloading IMDB dataset from Hugging Face ...")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from e

    ds = load_dataset("stanfordnlp/imdb")
    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()
    df = pd.concat([train, test], ignore_index=True)
    df = df.rename(columns={"text": "review", "label": "sentiment"})
    df["sentiment"] = df["sentiment"].map({0: "negative", 1: "positive"})
    df = df[["review", "sentiment"]]
    IMDB_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(IMDB_CSV, index=False)
    print(f"  saved {len(df):,} rows -> {IMDB_CSV}")


def download_glove():
    if GLOVE_TXT.exists():
        print(f"[skip] GloVe already at {GLOVE_TXT}")
        return
    print(f"Downloading GloVe (~822 MB zip) from {GLOVE_URL} ...")
    GLOVE_TXT.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(GLOVE_URL) as resp:
        buf = io.BytesIO(resp.read())
    print("  extracting glove.6B.100d.txt ...")
    with zipfile.ZipFile(buf) as zf, zf.open(GLOVE_MEMBER) as src, GLOVE_TXT.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  saved -> {GLOVE_TXT}")


def main():
    parser = argparse.ArgumentParser(description="Download IMDB + GloVe.")
    parser.add_argument("--imdb-only", action="store_true")
    parser.add_argument("--glove-only", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.glove_only:
        download_imdb()
    if not args.imdb_only:
        download_glove()
    print("\nDone. Next: python scripts/train_model.py")


if __name__ == "__main__":
    main()
