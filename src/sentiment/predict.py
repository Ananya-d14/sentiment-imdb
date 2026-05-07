import json
import pickle
from dataclasses import dataclass
from functools import lru_cache

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment.config import (
    CNN_MODEL_PATH,
    LABEL_NAMES,
    LSTM_MODEL_PATH,
    MAX_SEQUENCE_LEN,
    METADATA_PATH,
    NN_MODEL_PATH,
    TOKENIZER_PATH,
)
from sentiment.preprocessing import clean_text

_MODEL_PATHS = {
    "nn": NN_MODEL_PATH,
    "cnn": CNN_MODEL_PATH,
    "lstm": LSTM_MODEL_PATH,
}


@dataclass
class Prediction:
    label: str
    score: float
    probability_positive: float
    model_name: str

    def to_dict(self):
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "probability_positive": round(self.probability_positive, 4),
            "model_name": self.model_name,
        }


class SentimentPredictor:
    def __init__(self, model_name="lstm", model_path=None,
                 tokenizer_path=TOKENIZER_PATH):
        if model_name not in _MODEL_PATHS:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from {list(_MODEL_PATHS)}."
            )
        self.model_name = model_name
        self._model_path = model_path or _MODEL_PATHS[model_name]
        self._tokenizer_path = tokenizer_path
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self._model_path}. "
                    "Train it first with: python scripts/train_model.py"
                )
            self._model = load_model(self._model_path)
        if self._tokenizer is None:
            if not self._tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer not found at {self._tokenizer_path}. "
                    "Train first with: python scripts/train_model.py"
                )
            with self._tokenizer_path.open("rb") as fh:
                self._tokenizer = pickle.load(fh)

    def predict(self, text):
        return self.predict_batch([text])[0]

    def predict_batch(self, texts):
        self._load()
        cleaned = [clean_text(t) for t in texts]
        sequences = self._tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(sequences, padding="post", maxlen=MAX_SEQUENCE_LEN)
        probs = self._model.predict(padded, verbose=0).reshape(-1)
        results = []
        for p in probs:
            p = float(p)
            label_idx = int(p >= 0.5)
            confidence = p if label_idx == 1 else 1.0 - p
            results.append(Prediction(
                label=LABEL_NAMES[label_idx],
                score=confidence,
                probability_positive=p,
                model_name=self.model_name,
            ))
        return results


@lru_cache(maxsize=4)
def get_predictor(model_name="lstm"):
    return SentimentPredictor(model_name=model_name)


def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}
