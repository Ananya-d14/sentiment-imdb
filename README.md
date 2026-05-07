---
title: IMDB Sentiment Analyzer
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# IMDB Sentiment Analyzer

A small web app that reads a movie review and tells you whether it
sounds positive or negative.

**Try it live (no setup):**
[huggingface.co/spaces/ananyajoshids/sentiment-imdb](https://huggingface.co/spaces/ananyajoshids/sentiment-imdb)

Paste any movie review, hit *Analyze*, and you'll get a sentiment
label, a confidence score, and the cleaned-up text the model actually
saw. There are also three example reviews in the sidebar so you can
poke around without typing anything.

---

## What this is, in plain English

If I show you a movie review like:

> "Painfully boring and predictable. The dialogue is wooden and the plot
> makes no sense."

it's pretty obvious that's a negative review. But how would a computer
figure that out? That's the whole point of this project.

For this project I trained three different deep-learning models to
do that classification, on a dataset of 50,000 real IMDB reviews. The
best model (an LSTM) gets it right about **86% of the time** on
reviews it has never seen before.

## Why I built this

It started as a coursework Jupyter notebook (you can see the original
under `notebooks/`). Notebooks are great for exploration, but they're
not something you can hand to a friend and say "try this." So I
rewrote it into a proper Python package with a real UI, fixed a few
rough edges, wrote tests, containerized it, and shipped it to
Hugging Face Spaces. That's what's in this repo.

## How it works (the short version)

When you give the app a review, this is what happens:

1. **Clean the text.** Lowercase it, strip out HTML, drop digits and
   punctuation, remove common English stopwords like "the" and "is".
2. **Turn words into numbers.** Each unique word gets an integer ID,
   and the review becomes a list of those IDs (capped at 100 words).
3. **Look up word meanings.** Each ID is mapped to a 100-dimensional
   vector from **GloVe** (a famous pre-trained set of word embeddings
   from Stanford). Similar words have similar vectors.
4. **Run it through a neural network.** I tried three different
   architectures (a simple Dense layer, a 1-D CNN, and an LSTM). Each
   reads the review and outputs a single number between 0 and 1.
5. **Decide.** Above 0.5 is positive, below is negative. The closer to
   0 or 1, the more confident.

Steps 1-3 are the same for every model; only step 4 (the brain)
changes.

## The models

I tried all three on the same data and the same train/test split:

| Model | Test Accuracy | Notes |
|------|--------------:|-------|
| Dense (no recurrence) | 74.46% | Fast, but misses word order entirely |
| CNN (1-D convolution) | 84.76% | Picks up local n-gram patterns |
| **LSTM** | **85.84%** | Reads sequentially, default for the demo |

The LSTM wins, and it's what the live demo uses by default. You can
switch between them in the sidebar if you train all three locally.

## How to run it on your own machine

If you just want to see it work, the live demo is easier. If you want
to retrain or tinker with the code, here's how to get it running.

### Step 1: Get the code

```bash
git clone https://github.com/Ananya-d14/sentiment-imdb.git
cd sentiment-imdb
```

### Step 2: Set up a virtual environment

A virtual environment keeps the project's Python packages separate
from the rest of your system, so installing `tensorflow` here doesn't
mess with anything else.

```bash
python -m venv .venv

# activate it:
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

### Step 3: Install the dependencies

```bash
pip install -r requirements-dev.txt
```

This pulls in TensorFlow, Streamlit, NLTK, scikit-learn, and a few
others. It's the slowest step (about 2-5 minutes depending on your
connection).

### Step 4: Download the data

The IMDB dataset (~80 MB) and the GloVe embeddings (~1 GB zipped) are
too big to keep in the repo, so there's a script that fetches them:

```bash
python scripts/download_data.py
```

The IMDB CSV comes from Hugging Face. The GloVe file comes from
Stanford. Both end up under `data/raw/`.

### Step 5: Train the models

```bash
python scripts/train_model.py --no-early-stop
```

This trains all three models in turn and saves them under `models/`.
On a recent laptop CPU it takes about 3.5 minutes total (most of that
is the LSTM). On a GPU it's a few seconds.

If you only care about the LSTM:

```bash
python scripts/train_model.py --models lstm --no-early-stop
```

### Step 6: Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. You should see the same
UI as the live demo.

### Optional: Run the tests

```bash
pytest
```

There are 13 tests covering the text cleaning, the prediction
pipeline, and basic imports. They should all pass.

## Repo layout

```
app.py                        the Streamlit UI
src/sentiment/                the actual library code
  config.py                   paths and hyperparameters
  preprocessing.py            text cleaning (used in train and inference)
  data.py                     IMDB loading + train/test split
  embeddings.py               GloVe parser
  models.py                   the three Keras models
  train.py                    training loop, saves model + tokenizer
  predict.py                  the SentimentPredictor class
scripts/
  download_data.py            fetches IMDB + GloVe
  train_model.py              CLI for training
tests/                        pytest suite
notebooks/                    the original exploration notebook
data/samples/                 a few example reviews + their predictions
docs/                         model card and architecture notes
deploy/                       notes for HF Spaces and Docker
Dockerfile, docker-compose.yml
requirements.txt              what the app needs at runtime
requirements-dev.txt          everything else (training, tests, lint)
```

## Notes from building this

A few things that took some figuring out:

- **The original notebook saved the trained model but not the
  tokenizer.** That's a real problem, because at inference time you
  need to map words to the same integer IDs the model was trained on.
  Without the tokenizer, new text gets tokenized against a
  different vocabulary and predictions are nonsense. The training
  pipeline here pickles the tokenizer next to the model so they stay
  paired.
- **Early stopping was too aggressive at first.** I had it set to
  `patience=2` on val_accuracy, which killed the LSTM after about
  two epochs and dropped accuracy to ~80%. Letting it train the full
  10 epochs (`--no-early-stop`) recovered the 85.8% I expected. I
  left the early-stop callback in the code but disabled by default.
- **Stopword removal hurts on negation.** Words like "not" and
  "never" are in the NLTK English stopword list, which is bad for
  sentiment analysis. I kept the same preprocessing as the original
  notebook for consistency, but a v2 should keep negation words.

## Deploying it elsewhere

If you want to host this somewhere other than Hugging Face Spaces:

- See `deploy/HUGGINGFACE.md` for the HF flow (Docker SDK).
- See `deploy/DOCKER.md` for any container host (Render, Railway,
  Fly.io, Cloud Run, etc).

The `Dockerfile` is small and self-contained; if you `docker build`
and `docker run -p 8501:8501` it locally, you'll get the same app.

## References

- *Learning Word Vectors for Sentiment Analysis* by Maas et al., 2011
  (the IMDB dataset)
- *GloVe: Global Vectors for Word Representation* by Pennington,
  Socher, Manning, 2014

## License

MIT. See [LICENSE](LICENSE).

---

Author: Ananya Joshi.
Live demo: <https://huggingface.co/spaces/ananyajoshids/sentiment-imdb>.
Code: <https://github.com/Ananya-d14/sentiment-imdb>.
