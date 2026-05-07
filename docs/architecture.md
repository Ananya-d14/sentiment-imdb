# Architecture

## Data flow

```
       IMDB CSV (50k reviews)        GloVe.6B.100d
              |                           |
              v                           |
         clean_text                       |
              |                           |
              v                           |
     Keras Tokenizer (fit on train)       |
              |  |                        |
              |  +--> tokenizer.pkl       |
              v                           |
     pad_sequences (maxlen=100)           |
              |                           v
              |                  build_embedding_matrix
              |                           |
              v                           v
         +------------------------------------+
         |  Embedding (frozen GloVe-100d)     |
         +-----------------+------------------+
                           |
                           v
                +----------+----------+
                | NN | CNN | LSTM-128 |  (one model selected)
                +----------+----------+
                           |
                           v
                       sigmoid
                           |
                           v
                    P(positive)
```

At inference the same `clean_text` runs, the saved tokenizer maps tokens
to ints, and the chosen model produces a probability.

## Modules

| File | Purpose |
|---|---|
| `config.py` | paths, hyperparameters |
| `preprocessing.py` | `clean_text`, used in train and inference |
| `data.py` | IMDB load + 80/20 split |
| `embeddings.py` | GloVe parser, embedding matrix builder |
| `models.py` | Keras Sequential builders for NN, CNN, LSTM |
| `train.py` | training loop, saves tokenizer + model + metadata |
| `predict.py` | `SentimentPredictor` class for inference |
| `app.py` | Streamlit UI |
| `scripts/download_data.py` | fetch IMDB and GloVe |
| `scripts/train_model.py` | CLI wrapping train_all |

## Why the tokenizer is saved

The original notebook saved the trained Keras model but not the fitted
`Tokenizer`. At inference time on a fresh process, you'd need to refit a
tokenizer on... what? You don't have the training data anymore. And even
if you reconstructed it, the integer mapping would be different and the
saved embedding matrix would point at the wrong words.

The training pipeline here pickles the tokenizer to
`models/tokenizer.pkl` right after fitting. `SentimentPredictor` loads
both files together, so a fresh process can serve predictions that
match what the model was trained on.
