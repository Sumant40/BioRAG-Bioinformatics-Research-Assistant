# retrieval/reranker.py
from sentence_transformers import CrossEncoder


class BioReranker:
    """
    Cross-encoder re-ranker fine-tuned on MS MARCO.
    More expensive than bi-encoder but dramatically
    improves precision on ambiguous biomedical queries.
    """

    def __init__(self, top_k: int = 4):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_k = top_k

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        if not hits:
            return []

        pairs = [(query, h["text"]) for h in hits]
        scores = self.model.predict(pairs)

        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)

        reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[: self.top_k]