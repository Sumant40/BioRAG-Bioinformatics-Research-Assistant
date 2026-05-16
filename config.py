# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama-3.3-70b-versatile"

    EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
    EMBEDDING_DIMENSION = 768

    # Use /tmp for cloud (ephemeral) or a mounted disk path
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
    COLLECTION_NAME = "bio_knowledge_base"

    TOP_K = 8
    RERANK_TOP_K = 4
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64

    PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    UNIPROT_BASE_URL = "https://rest.uniprot.org"
    NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")