import re
from functools import lru_cache

_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z]")
_SINGLE_CHAR_RE = re.compile(r"\s+[a-zA-Z]\s+")
_MULTI_SPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _stopword_pattern():
    import nltk
    from nltk.corpus import stopwords

    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return re.compile(r"\b(" + r"|".join(words) + r")\b\s*")


def clean_text(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = _TAG_RE.sub("", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _SINGLE_CHAR_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _stopword_pattern().sub("", text)
    return text.strip()


def clean_corpus(texts):
    return [clean_text(t) for t in texts]
