
# ingestion/embedder.py
import os
import numpy as np
import requests

class BioEmbedder:
    """
    Uses Groq-compatible or HuggingFace Inference API for embeddings.
    Zero local model RAM — embeddings computed via API call.
    """

    def __init__(self, model_name: str = None):
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}",
            "Content-Type": "application/json"
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        # HuggingFace free inference API — batch in groups of 32
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=60
            )
            result = response.json()
            if isinstance(result, list):
                all_embeddings.extend(result)
            else:
                raise ValueError(f"HF API error: {result}")
        arr = np.array(all_embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-8)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]