# retrieval/reranker.py
# retrieval/reranker.py

class BioReranker:
    """
    Lightweight reranker using hybrid score — no local model needed.
    Saves ~200MB RAM vs cross-encoder.
    """

    def __init__(self, top_k: int = 4):
        self.top_k = top_k

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        if not hits:
            return []

        query_terms = set(query.lower().split())

        for hit in hits:
            text_lower = hit["text"].lower()
            # Boost score if query terms appear in text
            term_overlap = sum(1 for t in query_terms if t in text_lower)
            overlap_score = term_overlap / (len(query_terms) + 1)
            base_score = hit.get("hybrid_score", hit.get("score", 0.5))
            hit["rerank_score"] = 0.7 * base_score + 0.3 * overlap_score

        reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[: self.top_k]