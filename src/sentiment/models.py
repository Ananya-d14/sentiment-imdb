from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Embedding,
    Flatten,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Sequential

from sentiment.config import EMBEDDING_DIM, MAX_SEQUENCE_LEN


def _embedding_layer(vocab_size, embedding_matrix):
    return Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LEN,
        trainable=False,
    )


def build_nn(vocab_size, embedding_matrix):
    model = Sequential(name="simple_nn")
    model.add(_embedding_layer(vocab_size, embedding_matrix))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(vocab_size, embedding_matrix):
    model = Sequential(name="cnn")
    model.add(_embedding_layer(vocab_size, embedding_matrix))
    model.add(Conv1D(128, 5, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(vocab_size, embedding_matrix):
    model = Sequential(name="lstm")
    model.add(_embedding_layer(vocab_size, embedding_matrix))
    model.add(LSTM(128))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


BUILDERS = {
    "nn": build_nn,
    "cnn": build_cnn,
    "lstm": build_lstm,
}
