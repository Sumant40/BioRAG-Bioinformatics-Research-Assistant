# retrieval/vector_store.py
# retrieval/vector_store.py
import chromadb
from chromadb.config import Settings
import uuid
import json
import shutil
import os
from ingestion.chunker import Chunk


def sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, dict):
            clean[key] = value.get("#text", json.dumps(value))
        elif isinstance(value, list):
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
        self.persist_dir = config.CHROMA_PERSIST_DIR

        self.client = self._create_client()
        self.collection = self._create_collection(config.COLLECTION_NAME)

    def _create_client(self):
        try:
            client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            # Test the client works
            client.list_collections()
            return client
        except Exception as e:
            print(f"ChromaDB corrupted ({e}) — wiping and recreating...")
            # Delete corrupted DB and start fresh
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
            return chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

    def _create_collection(self, name: str):
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"Collection error ({e}) — deleting and recreating collection...")
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
            return self.client.create_collection(
                name=name,
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
            metadatas=[sanitize_metadata(c.metadata) for c in chunks],
        )
        print(f"Added {len(chunks)} chunks to vector store.")

    def similarity_search(self, query: str, top_k: int = 8) -> list[dict]:
        query_embedding = self.embedder.embed_query(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
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
        try:
            return self.collection.count()
        except Exception:
            return 0