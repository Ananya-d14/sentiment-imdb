import numpy as np

from sentiment.config import EMBEDDING_DIM, GLOVE_TXT


def load_glove(glove_path=GLOVE_TXT, dim=EMBEDDING_DIM):
    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found at {glove_path}. "
            "Run scripts/download_data.py first."
        )
    embeddings = {}
    with glove_path.open(encoding="utf8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] != dim:
                continue
            embeddings[word] = vec
    return embeddings


def build_embedding_matrix(word_index, embeddings, dim=EMBEDDING_DIM):
    vocab_size = len(word_index) + 1
    matrix = np.zeros((vocab_size, dim), dtype=np.float32)
    for word, idx in word_index.items():
        vec = embeddings.get(word)
        if vec is not None:
            matrix[idx] = vec
    return matrix
