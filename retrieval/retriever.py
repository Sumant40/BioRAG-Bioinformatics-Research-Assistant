# retrieval/retriever.py
from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    """
    Combines dense (vector) + sparse (BM25) retrieval.
    Critical for biomedical queries with exact gene names,
    accession IDs, or drug compound names where BM25 excels.
    """

    def __init__(self, vector_store, config):
        self.vector_store = vector_store
        self.top_k = config.TOP_K
        self._bm25 = None
        self._corpus = []   # populated at query time from vector store snapshot

    def _build_bm25(self, docs: list[str]):
        tokenized = [d.lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus = docs

    def retrieve(self, query: str, alpha: float = 0.6) -> list[dict]:
        """
        alpha: weight for dense scores (1-alpha for BM25).
        Higher alpha → prefer semantic similarity.
        Lower alpha → prefer exact keyword match.
        """
        # Dense retrieval
        dense_hits = self.vector_store.similarity_search(query, top_k=self.top_k * 2)

        if not dense_hits:
            return []

        # BM25 over the dense candidate set (efficient — not full corpus)
        candidate_texts = [h["text"] for h in dense_hits]
        self._build_bm25(candidate_texts)
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_scores_norm = bm25_scores / (bm25_scores.max() + 1e-8)

        # Fuse scores
        for i, hit in enumerate(dense_hits):
            hit["hybrid_score"] = alpha * hit["score"] + (1 - alpha) * bm25_scores_norm[i]

        # Sort by hybrid score and return top_k
        ranked = sorted(dense_hits, key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[: self.top_k]