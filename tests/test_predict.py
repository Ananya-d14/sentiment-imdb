import pytest

from sentiment.config import LSTM_MODEL_PATH, TOKENIZER_PATH
from sentiment.predict import Prediction, SentimentPredictor

pytestmark = pytest.mark.skipif(
    not (LSTM_MODEL_PATH.exists() and TOKENIZER_PATH.exists()),
    reason="Trained model or tokenizer not found; run scripts/train_model.py first.",
)


def test_predictor_returns_valid_prediction():
    predictor = SentimentPredictor(model_name="lstm")
    out = predictor.predict("This was an absolute masterpiece, I loved every minute.")
    assert isinstance(out, Prediction)
    assert out.label in {"positive", "negative"}
    assert 0.0 <= out.score <= 1.0
    assert 0.0 <= out.probability_positive <= 1.0


def test_batch_returns_one_per_input():
    predictor = SentimentPredictor(model_name="lstm")
    texts = ["Loved it.", "Hated it.", "It was fine."]
    out = predictor.predict_batch(texts)
    assert len(out) == len(texts)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        SentimentPredictor(model_name="bert-xxl")
