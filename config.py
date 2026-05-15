# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL = "llama-3.3-70b-versatile"   # best free model on Groq

    # Everything else stays exactly the same
    EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
    EMBEDDING_DIMENSION = 768
    CHROMA_PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "bio_knowledge_base"
    TOP_K = 8
    RERANK_TOP_K = 4
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    UNIPROT_BASE_URL = "https://rest.uniprot.org"
    NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")