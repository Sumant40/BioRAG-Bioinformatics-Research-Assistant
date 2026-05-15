# api/main.py
from fastapi import FastAPI, HTTPException
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
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="BioRAG API", version="1.0.0")
config = Config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise components (singleton pattern)
embedder = BioEmbedder(config.EMBEDDING_MODEL)
vector_store = BioVectorStore(config, embedder)
retriever = HybridRetriever(vector_store, config)
reranker = BioReranker(top_k=config.RERANK_TOP_K)
generator = BioGenerator(config)
agent = BioAgent(retriever, reranker, generator, config)
chunker = BioChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
pubmed_loader = PubMedLoader(config)


class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    pubmed_query: str
    max_results: int = 100


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = agent.run(req.query)
    return result


@app.post("/ingest/pubmed")
async def ingest_pubmed(req: IngestRequest):
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
    return {"status": "ok", "chunks_indexed": vector_store.count()}