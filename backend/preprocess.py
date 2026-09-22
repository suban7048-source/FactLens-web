"""
preprocess.py — Text cleaning and NLP preprocessing utilities
for the Fake News Detection System.
"""

import re
import nltk

# Download required NLTK data silently
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'we', 'you',
        'he', 'she', 'they', 'me', 'us', 'him', 'her', 'them', 'my',
        'your', 'his', 'our', 'their', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'not', 'no', 'so', 'if', 'as',
    }

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    def _stem(word):
        return _stemmer.stem(word)
except Exception:
    def _stem(word):
        return word


def preprocess_text(text: str) -> str:
    """
    Clean and normalize news text for ML classification.

    Steps:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove special characters and digits
        5. Collapse whitespace
        6. Remove stopwords
        7. Apply Porter Stemming

    Returns:
        Cleaned, stemmed token string ready for TF-IDF vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # 5. Remove special characters and digits (keep only letters)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 6. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Tokenize, remove stopwords, apply stemming
    tokens = [
        _stem(token)
        for token in text.split()
        if token not in STOP_WORDS and len(token) > 2
    ]

    return ' '.join(tokens)


def preprocess_batch(texts) -> list:
    """Preprocess a list of texts. Returns list of cleaned strings."""
    return [preprocess_text(t) for t in texts]
