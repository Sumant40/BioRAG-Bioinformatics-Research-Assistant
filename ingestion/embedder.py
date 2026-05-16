
# ingestion/embedder.py
import numpy as np
import math
from collections import Counter


class BioEmbedder:
    """
    TF-IDF based embedder — runs fully in memory.
    No external API, no HuggingFace token, no RAM issues.
    Vocabulary is built from ingested documents.
    """

    def __init__(self, model_name: str = None):
        self.vocab = {}
        self.idf = {}
        self.dim = 512
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        # Simple whitespace + punctuation tokenizer
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z][a-z0-9\-]{1,20}\b', text)
        return tokens

    def _build_vocab(self, texts: list[str]):
        # Build vocabulary from all texts
        doc_freq = Counter()
        tokenized = []
        for text in texts:
            tokens = set(self._tokenize(text))
            doc_freq.update(tokens)
            tokenized.append(tokens)

        # Keep top-dim most frequent terms
        top_terms = [term for term, _ in doc_freq.most_common(self.dim)]
        self.vocab = {term: i for i, term in enumerate(top_terms)}

        # Compute IDF
        n_docs = len(texts)
        self.idf = {}
        for term, idx in self.vocab.items():
            df = doc_freq[term]
            self.idf[term] = math.log((n_docs + 1) / (df + 1)) + 1

        self._fitted = True

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        token_counts = Counter(tokens)
        total = len(tokens) + 1

        vec = np.zeros(self.dim, dtype=np.float32)
        for token, count in token_counts.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tf = count / total
                idf = self.idf.get(token, 1.0)
                vec[idx] = tf * idf

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        # Fit vocab if not fitted or if we have new docs
        if not self._fitted:
            self._build_vocab(texts)

        vectors = np.array([self._vectorize(t) for t in texts], dtype=np.float32)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            # Fallback — build minimal vocab from query itself
            self._build_vocab([query])
        return self._vectorize(query)