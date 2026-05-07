# Model card

## Overview

Binary sentiment classifier for English movie reviews (positive vs.
negative). Three classifier heads are trained on top of frozen 100-d
GloVe embeddings; the LSTM head is the default for serving.

| Field | Value |
|---|---|
| Task | Binary text classification |
| Languages | English |
| Inputs | Free-text movie reviews (truncated/padded to 100 tokens after cleaning) |
| Outputs | label (positive / negative), probability_positive, score |
| Training data | IMDB Large Movie Review Dataset (50,000 reviews, balanced) |
| Embeddings | GloVe `glove.6B.100d` (frozen) |
| Library | TensorFlow / Keras 2.15+ |
| License | MIT |

## Training data

- Source: `stanfordnlp/imdb` on Hugging Face (Maas et al. 2011)
- Size: 50,000 reviews
- Balance: 50/50 positive/negative
- Split: shuffled 80/20 with `random_state=42`; train portion further
  uses 20% as validation

## Preprocessing

Identical at train and inference time (`clean_text` in
`src/sentiment/preprocessing.py`):

1. lowercase
2. strip HTML tags
3. drop non-alphabetic characters
4. drop single-character tokens
5. collapse whitespace
6. remove English stopwords (NLTK)

Then Keras `Tokenizer` (fit on the train split) maps tokens to ints and
sequences are padded to length 100.

## Architectures

| Model | Layers | Trainable params | Frozen params |
|---|---|---:|---:|
| NN | Embedding -> Flatten -> Dense(1) | 10,001 | 9,231,400 |
| CNN | Embedding -> Conv1D(128, 5) -> GlobalMaxPool -> Dense(1) | 64,257 | 9,231,400 |
| LSTM | Embedding -> LSTM(128) -> Dense(1) | 117,377 | 9,231,400 |

Activation: sigmoid. Loss: binary crossentropy. Optimizer: Adam.

## Performance

| Model | Test accuracy | Test loss |
|---|---:|---:|
| NN | 0.7446 | 0.6041 |
| CNN | 0.8476 | 0.3939 |
| LSTM | 0.8584 | 0.3467 |

## Limitations

- Trained on movie reviews only; accuracy on tweets, product reviews,
  or non-English text will be lower.
- Sarcasm and irony are hard for short-context models; expect failures.
- Anything past the first 100 cleaned tokens is truncated.
- Stopword removal drops words like "not", which hurts negated reviews.
  Inherited from the original notebook for parity.
- Out-of-vocabulary words become zero vectors.

## Notes from training

- The first run used `EarlyStopping(patience=2)`. It stopped the LSTM
  after about 2 epochs and test accuracy fell to ~80%. Running the full
  10 epochs (`--no-early-stop`) brings it back to ~85.8%. This is the
  configuration that produced the artifacts shipped here.
- Vocabulary size with this seed is 92,314 (the original notebook
  reported 92,394; small differences come from the dataset version on
  HuggingFace vs. the original CSV).
- `metadata.json` in `models/` records the per-epoch accuracy/loss
  history for each model, so you can plot the training curves without
  retraining.

## Reproducing results

```bash
python scripts/download_data.py
python scripts/train_model.py --no-early-stop
```

The data split is seeded with `RANDOM_STATE=42` in
`src/sentiment/config.py`. Keras-internal randomness is not seeded so
final test accuracy typically falls within +/- 0.5%.
