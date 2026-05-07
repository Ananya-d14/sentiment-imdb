---
title: IMDB Sentiment Analyzer
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# IMDB Sentiment Analyzer

Sentiment classification on IMDB movie reviews. Three models (Dense, CNN,
LSTM) trained on 50K reviews with GloVe word embeddings; the LSTM hits
85.8% test accuracy and is what the Streamlit demo serves by default.

This started as a coursework Jupyter notebook (kept under `notebooks/`).
I rewrote it into a packaged Python app with a training CLI, a Streamlit
UI, tests, and a Docker build so it could actually be deployed and used,
not just shown.

## Models

| Model | Test Accuracy | Test Loss |
|------|--------------:|----------:|
| Dense NN | 74.46% | 0.6041 |
| CNN | 84.76% | 0.3939 |
| LSTM | **85.84%** | **0.3467** |

Trained on CPU. The full run (all three models, 10 epochs each, no
early stopping) takes about 3.5 min on a recent Intel laptop. The LSTM
is the slowest at ~3 min on its own.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Train

The IMDB dataset (~80 MB) and GloVe embeddings (~1 GB zip) aren't in the
repo. Download them, then train:

```bash
python scripts/download_data.py
python scripts/train_model.py --no-early-stop          # all three models
python scripts/train_model.py --models lstm --no-early-stop   # just the LSTM
```

This saves `models/{nn,cnn,lstm}_model.keras`, `models/tokenizer.pkl`,
and `models/metadata.json`.

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t sentiment-imdb .
docker run -p 8501:8501 -v $(pwd)/models:/app/models:ro sentiment-imdb
```

Or `docker compose up`.

## Project layout

```
app.py                        # streamlit demo
src/sentiment/                # library code
  config.py                   # paths + hyperparams
  preprocessing.py            # text cleaning
  data.py                     # IMDB load + split
  embeddings.py               # GloVe loader
  models.py                   # NN / CNN / LSTM
  train.py                    # training pipeline
  predict.py                  # inference
scripts/
  download_data.py
  train_model.py
tests/                        # pytest
notebooks/                    # original exploration
data/samples/                 # sample reviews + predictions
docs/                         # model card, architecture
deploy/                       # HF Spaces and Docker notes
```

## Pipeline

```
raw review
  -> clean text   (strip HTML, lowercase, drop stopwords/digits)
  -> tokenize     (Keras Tokenizer, fit on training split)
  -> pad to 100 tokens
  -> embedding    (GloVe.6B.100d, frozen)
  -> classifier   (Dense | Conv1D | LSTM-128)
  -> sigmoid      -> P(positive)
```

The same `clean_text` function is used at training and at inference, so
the two paths apply the same normalization.

## Notes

- The original notebook saved the trained model but not the fitted
  `Tokenizer`. Without the tokenizer, new text gets mapped to a
  different vocabulary and predictions are garbage. The training
  pipeline here pickles the tokenizer alongside the model.
- I left `EarlyStopping` available in the training code but the default
  patience of 2 was too aggressive for the LSTM. It stopped after a
  couple of epochs and accuracy dropped to ~80%. Running the full 10
  epochs gets the LSTM to ~85.8% as reported above. Pass
  `--no-early-stop` to reproduce.
- Stopword removal drops words like "not", which hurts on negated
  reviews. Kept it for parity with the original notebook; would
  reconsider for a v2.

## Deployment

See [deploy/HUGGINGFACE.md](deploy/HUGGINGFACE.md) for Hugging Face
Spaces, or [deploy/DOCKER.md](deploy/DOCKER.md) for any container host.

## References

- IMDB dataset: Maas et al., 2011
- GloVe: Pennington, Socher, Manning, 2014

## License

MIT. See [LICENSE](LICENSE).

Author: Ananya Joshi.
