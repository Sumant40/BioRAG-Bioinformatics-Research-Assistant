# api/main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import Config
from ingestion.loaders import PubMedLoader
from ingestion.chunker import BioChunker
from ingestion.embedder import BioEmbedder
from retrieval.vector_store import BioVectorStore
from retrieval.retriever import HybridRetriever
from retrieval.reranker import BioReranker
from generation.generator import BioGenerator
from agents.bio_agent import BioAgent

app = FastAPI(title="BioRAG API", version="1.0.0")
config = Config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy globals — initialized on first request, not at startup
_embedder = None
_vector_store = None
_retriever = None
_reranker = None
_generator = None
_agent = None
_chunker = None
_pubmed_loader = None

def get_components():
    global _embedder, _vector_store, _retriever, _reranker
    global _generator, _agent, _chunker, _pubmed_loader

    if _agent is None:
        _embedder = BioEmbedder(config.EMBEDDING_MODEL)
        _vector_store = BioVectorStore(config, _embedder)
        _retriever = HybridRetriever(_vector_store, config)
        _reranker = BioReranker(top_k=config.RERANK_TOP_K)
        _generator = BioGenerator(config)
        _agent = BioAgent(_retriever, _reranker, _generator, config)
        _chunker = BioChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        _pubmed_loader = PubMedLoader(config)

    return _agent, _chunker, _pubmed_loader, _vector_store


class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    pubmed_query: str
    max_results: int = 25


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    agent, _, _, _ = get_components()
    result = agent.run(req.query)
    return result


@app.post("/ingest/pubmed")
async def ingest_pubmed(req: IngestRequest):
    agent, chunker, pubmed_loader, vector_store = get_components()
    records = pubmed_loader.search(req.pubmed_query, max_results=req.max_results)
    chunks = chunker.chunk_records(records, text_field="abstract")
    vector_store.add_chunks(chunks)
    return {
        "ingested_records": len(records),
        "chunks_added": len(chunks),
        "total_chunks": vector_store.count(),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": _vector_store.count() if _vector_store else 0}