# ingestion/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np


class BioEmbedder:
    """
    Uses BioBERT fine-tuned on biomedical NLI tasks —
    significantly outperforms general-purpose embeddings
    on gene/disease/pathway similarity tasks.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine similarity via dot product
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )