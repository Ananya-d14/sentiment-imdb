import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment.config import IMDB_CSV, RANDOM_STATE, TEST_SPLIT
from sentiment.preprocessing import clean_corpus


def load_raw(csv_path=IMDB_CSV):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"IMDB dataset not found at {csv_path}. "
            "Run scripts/download_data.py first."
        )
    df = pd.read_csv(csv_path)
    if {"review", "sentiment"} - set(df.columns):
        raise ValueError(
            f"Expected columns 'review' and 'sentiment', got {list(df.columns)}"
        )
    return df


def load_split(csv_path=IMDB_CSV, test_size=TEST_SPLIT, random_state=RANDOM_STATE):
    df = load_raw(csv_path)
    x = clean_corpus(df["review"].tolist())
    y = np.array([1 if s == "positive" else 0 for s in df["sentiment"]], dtype=np.int32)
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
