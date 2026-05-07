import json
import pickle
import time
from dataclasses import asdict, dataclass

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sentiment.config import (
    BATCH_SIZE,
    CNN_MODEL_PATH,
    EPOCHS,
    LSTM_MODEL_PATH,
    MAX_SEQUENCE_LEN,
    METADATA_PATH,
    MODELS_DIR,
    NN_MODEL_PATH,
    TOKENIZER_PATH,
    VALIDATION_SPLIT,
)
from sentiment.data import load_split
from sentiment.embeddings import build_embedding_matrix, load_glove
from sentiment.models import BUILDERS

_MODEL_PATHS = {
    "nn": NN_MODEL_PATH,
    "cnn": CNN_MODEL_PATH,
    "lstm": LSTM_MODEL_PATH,
}


@dataclass
class TrainResult:
    name: str
    test_accuracy: float
    test_loss: float
    train_seconds: float
    history: dict


def _fit_tokenizer(x_train):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    return tokenizer


def _to_sequences(tokenizer, texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, padding="post", maxlen=MAX_SEQUENCE_LEN)


def train_all(epochs=EPOCHS, batch_size=BATCH_SIZE, early_stop=True,
              which=("nn", "cnn", "lstm")):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and splitting IMDB data ...")
    x_train, x_test, y_train, y_test = load_split()

    print("Fitting tokenizer ...")
    tokenizer = _fit_tokenizer(x_train)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"  vocab size: {vocab_size}")

    x_train_seq = _to_sequences(tokenizer, x_train)
    x_test_seq = _to_sequences(tokenizer, x_test)

    print("Loading GloVe ...")
    glove = load_glove()
    embedding_matrix = build_embedding_matrix(tokenizer.word_index, glove)

    # Save the fitted tokenizer so inference uses the same vocab mapping
    # as training. Without this, predict.py can't reproduce the integer
    # encoding the model was trained on.
    with TOKENIZER_PATH.open("wb") as fh:
        pickle.dump(tokenizer, fh)
    print(f"Saved tokenizer to {TOKENIZER_PATH}")

    callbacks = []
    if early_stop:
        callbacks.append(EarlyStopping(patience=2, restore_best_weights=True,
                                       monitor="val_accuracy", mode="max"))

    results = {}
    for name in which:
        if name not in BUILDERS:
            raise ValueError(f"Unknown model: {name}")
        print(f"\n=== Training {name.upper()} ===")
        model = BUILDERS[name](vocab_size, embedding_matrix)
        model.summary()
        start = time.time()
        history = model.fit(
            x_train_seq, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
        )
        elapsed = time.time() - start
        loss, accuracy = model.evaluate(x_test_seq, y_test, verbose=0)
        path = _MODEL_PATHS[name]
        model.save(path)
        print(f"  test accuracy: {accuracy:.4f}  saved to {path}")

        results[name] = TrainResult(
            name=name,
            test_accuracy=float(accuracy),
            test_loss=float(loss),
            train_seconds=float(elapsed),
            history={k: [float(v) for v in vs] for k, vs in history.history.items()},
        )

    metadata = {
        "vocab_size": vocab_size,
        "max_sequence_len": MAX_SEQUENCE_LEN,
        "models": {name: asdict(r) for name, r in results.items()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved metadata to {METADATA_PATH}")
    return results
