# retrieval/vector_store.py
import chromadb
from chromadb.config import Settings
import uuid
import json
from ingestion.chunker import Chunk


def sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB only accepts str, int, float, bool, or None as metadata values.
    This flattens any nested dicts/lists into JSON strings.
    """
    clean = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, dict):
            # Flatten dict — try to extract text content if it's a PubMed tag dict
            if "#text" in value:
                clean[key] = str(value["#text"])
            else:
                clean[key] = json.dumps(value)
        elif isinstance(value, list):
            # Convert list to comma-separated string
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(item.get("#text", json.dumps(item)))
                else:
                    parts.append(str(item))
            clean[key] = ", ".join(parts)
        else:
            clean[key] = str(value)
    return clean


class BioVectorStore:

    def __init__(self, config, embedder):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts).tolist()

        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[sanitize_metadata(c.metadata) for c in chunks],  # sanitized
        )
        print(f"Added {len(chunks)} chunks to vector store.")

    def similarity_search(self, query: str, top_k: int = 8) -> list[dict]:
        query_embedding = self.embedder.embed_query(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,
            })
        return hits

    def count(self) -> int:
        return self.collection.count()