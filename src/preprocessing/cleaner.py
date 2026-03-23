"""
Phase 2 -- Text cleaning and lemmatization.

Public API
----------
clean_text(text)              -> lowercased, stripped of noise
tokenize_and_lemmatize(text)  -> lemmas, no stopwords
full_preprocess(text)         -> clean_text -> tokenize_and_lemmatize
"""

import re
import string

import contractions
import spacy
from nltk.corpus import stopwords

# Load once at module import time
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

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """
    Normalisation steps (order matters):
    1. Lowercase
    2. Expand contractions (don't -> do not)
    3. Remove URLs
    4. Remove @mentions
    5. Remove #hashtags
    6. Strip non-ASCII characters
    7. Remove punctuation
    8. Collapse whitespace
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Lemmatize and remove stopwords / punctuation / very short tokens."""
    if not text:
        return ""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in STOP_WORDS
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
    ]
    return " ".join(tokens)


def full_preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize & lemmatize."""
    cleaned = clean_text(text)
    return tokenize_and_lemmatize(cleaned)
