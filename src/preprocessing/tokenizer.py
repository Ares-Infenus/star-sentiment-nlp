"""Lightweight tokenization utilities (used by embeddings)."""

import spacy
from nltk.corpus import stopwords

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    """Simple whitespace + spaCy tokenisation, no filtering."""
    return [token.text for token in nlp(text) if not token.is_space]


def tokenize_filtered(text: str) -> list[str]:
    """Tokens without stopwords, punctuation, or very short words."""
    return [
        token.lemma_
        for token in nlp(text)
        if token.text not in STOP_WORDS
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
    ]
