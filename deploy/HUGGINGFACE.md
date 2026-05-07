# Deploy to Hugging Face Spaces

The `README.md` already has the YAML frontmatter Spaces needs
(`sdk: streamlit`, `app_file: app.py`).

## Prerequisites

- Hugging Face account: https://huggingface.co/join
- `git` and `git-lfs`
- `pip install huggingface_hub`
- Trained artifacts in `models/` (tokenizer.pkl, *.keras, metadata.json)

## Steps

```bash
# log in
huggingface-cli login

# create the Space
huggingface-cli repo create sentiment-imdb --type space --space_sdk streamlit

# add the Space as a remote and push
git remote add space https://huggingface.co/spaces/<USERNAME>/sentiment-imdb
git push space main
```

First build takes ~5 minutes.

## Model weights

Trained `.keras` files are usually a few MB each. Push them via git-lfs:

```bash
git lfs install
git lfs track "models/*.keras"
git lfs track "models/*.pkl"
git add .gitattributes models/
git commit -m "Add trained weights"
git push space main
```

If weights get large, host them on the HF Model Hub and pull at runtime:

```python
from huggingface_hub import hf_hub_download
from sentiment.config import MODELS_DIR

REPO = "USERNAME/sentiment-imdb-weights"
for f in ("tokenizer.pkl", "lstm_model.keras", "metadata.json"):
    hf_hub_download(REPO, f, local_dir=MODELS_DIR)
```

## Updating

```bash
git push space main
```

Build and runtime logs are visible at
`huggingface.co/spaces/<USERNAME>/sentiment-imdb`.
