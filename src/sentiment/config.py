import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("SENTIMENT_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = Path(os.getenv("SENTIMENT_MODELS_DIR", PROJECT_ROOT / "models"))

IMDB_CSV = RAW_DIR / "IMDB_Dataset.csv"
GLOVE_TXT = RAW_DIR / "glove.6B.100d.txt"

TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
NN_MODEL_PATH = MODELS_DIR / "nn_model.keras"

MAX_SEQUENCE_LEN = 100
EMBEDDING_DIM = 100
TEST_SPLIT = 0.20
VALIDATION_SPLIT = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 128
EPOCHS = 10

LABEL_NAMES = ("negative", "positive")
