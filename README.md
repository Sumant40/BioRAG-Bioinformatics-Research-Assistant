# 🧬 BioRAG — Bioinformatics Research Assistant

> A production-ready Retrieval-Augmented Generation (RAG) system for bioinformatics research. Query PubMed literature, UniProt protein databases, and KEGG pathways using natural language — powered by an agentic AI pipeline with grounded, cited answers.

![BioRAG Frontend displaying a conversation with the BioRAG bioinformatics research assistant. The interface has a dark theme with a left sidebar showing ingestion options for PubMed search queries, max results selector set to 10 papers, and an Ingest Papers button. Quick queries are listed below including BRCA1 DNA repair mechanisms, EGFR mutations in lung cancer, CRISPR off-target effects, and tumor microenvironment topics. The main panel shows a researcher asking What is the role of BRCA1 in DNA repair? with BioRAG providing a detailed, citation-rich response explaining BRCA1's role in homologous recombination, genomic stability, and breast cancer development, with multiple PubMed IDs (PMIDs) linked inline. The response is professional and informative, conveying expertise and scientific rigor. At the bottom, a new query prompt asks about EGFR mutations in lung cancer treatment response. The interface shows 50 chunks indexed and connected status indicator in the top right corner.](results/image.png)

---

## 📌 What is BioRAG?

BioRAG is an end-to-end AI research assistant built for the life sciences industry. It combines three cutting-edge AI paradigms:

| Paradigm | Role in BioRAG |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Retrieves relevant biomedical literature chunks before generating answers, preventing hallucination |
| **GenAI** | Uses a large language model (LLaMA 3 via Groq) to synthesize grounded, citation-rich answers |
| **Agentic AI** | An orchestrator agent routes queries to live tools (PubMed, UniProt, KEGG) when the knowledge base needs supplementing |

**Industry relevance:** Pharma R&D, clinical genomics, diagnostics, agricultural biotech, drug safety, and any domain requiring fast, auditable answers from scientific literature.

---

## ✨ Features

- 🔍 **Hybrid retrieval** — Dense vector search (PubMed-BERT) + BM25 sparse retrieval for exact gene/drug name matching
- 🤖 **Agentic query routing** — Detects gene symbols, pathway IDs, and recency signals; triggers live API lookups automatically
- 📚 **Multi-source ingestion** — PubMed abstracts, UniProt protein annotations, KEGG pathways, local PDFs
- ✅ **Citation-grounded answers** — Every claim linked to a PMID or UniProt accession
- 🔁 **Cross-encoder re-ranking** — Re-ranks retrieved chunks for maximum precision before generation
- 🌐 **REST API** — FastAPI backend with auto-generated Swagger UI at `/docs`
- 🖥️ **Web frontend** — Dark-themed research interface with ingestion panel, quick queries, and source inspection

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│  PubMed │ UniProt │ KEGG │ PDFs │ Databases │ APIs     │
│              ↓ Chunker ↓ BioBERT Embedder               │
│                    Vector Store (ChromaDB)              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    RAG LAYER                            │
│   Query Transform → Hybrid Retrieval → Re-ranker        │
│              → Context Assembler                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  GENAI CORE                             │
│   Prompt Engine → LLM (LLaMA 3.3 via Groq)             │
│   → Grounding & Guardrails → Memory / Cache             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  AGENTIC LAYER                          │
│   Orchestrator: Plan → Act → Observe → Iterate          │
│   Tools: PubMed Live │ UniProt │ KEGG │ Sub-agents      │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
bio-rag/
├── .env                          # API keys (never commit this)
├── config.py                     # Central configuration
├── run.py                        # Reliable entry point
│
├── ingestion/
│   ├── loaders.py                # PubMed, UniProt, PDF loaders
│   ├── chunker.py                # Bio-aware text chunking
│   └── embedder.py               # PubMed-BERT embeddings
│
├── retrieval/
│   ├── vector_store.py           # ChromaDB interface
│   ├── retriever.py              # Hybrid BM25 + dense retrieval
│   └── reranker.py               # Cross-encoder re-ranking
│
├── generation/
│   ├── prompt_templates.py       # Biomedical system prompts
│   └── generator.py              # Groq LLM interface
│
├── agents/
│   ├── bio_agent.py              # Agentic query orchestrator
│   └── tools.py                  # Live API tools
│
├── api/
│   └── main.py                   # FastAPI app & endpoints
│
└── biorag-frontend.html          # Web UI (open in browser)
```

---

## 🚀 Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/yourname/bio-rag.git
cd bio-rag
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install langchain langchain-community chromadb sentence-transformers \
    biopython fastapi uvicorn pypdf requests xmltodict \
    rank-bm25 tiktoken python-dotenv groq
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
NCBI_API_KEY=your_ncbi_key_here
```

- **Groq API key** (free) → [console.groq.com](https://console.groq.com)
- **NCBI API key** (free) → [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) *(optional but raises PubMed rate limit)*

### 4. Start the server

```bash
python run.py
# or
uvicorn api.main:app --reload --port 8000
```

Server starts at `http://localhost:8000`
Swagger UI available at `http://localhost:8000/docs`

### 5. Open the frontend

Open `biorag-frontend.html` in your browser. The status indicator turns **green** when connected.

> ⚠️ **CORS fix required** — Add this to `api/main.py` after `app = FastAPI(...)`:
> ```python
> from fastapi.middleware.cors import CORSMiddleware
> app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
> ```

---

## 📡 API Reference

### `POST /ingest/pubmed`
Fetches papers from PubMed, chunks them, embeds with PubMed-BERT, and stores in ChromaDB.

```json
{
  "pubmed_query": "BRCA1 breast cancer DNA repair",
  "max_results": 50
}
```

**Response:**
```json
{
  "ingested_records": 50,
  "chunks_added": 134,
  "total_chunks": 134
}
```

---

### `POST /query`
Runs the full RAG + Agentic pipeline and returns a grounded answer.

```json
{
  "query": "What are the mechanisms by which BRCA1 mutations increase cancer risk?"
}
```

**Response:**
```json
{
  "answer": "BRCA1 mutations impair homologous recombination repair... [PMID: 12345]",
  "sources": [...],
  "usage": { "input_tokens": 820, "output_tokens": 310 },
  "intent": { "needs_live_pubmed": false, "gene_lookup": true, "pathway_query": false },
  "live_fetch": true
}
```

---

### `GET /health`
Returns server status and number of indexed chunks.

```json
{
  "status": "ok",
  "chunks_indexed": 134
}
```

---

## 💡 Example Queries

```
What are the mechanisms by which BRCA1 mutations increase cancer risk?
How do EGFR mutations affect response to targeted therapy in NSCLC?
What is the role of TP53 in apoptosis regulation?
Explain CRISPR-Cas9 off-target effects in clinical gene therapy trials.
What genes are associated with Lynch syndrome and hereditary colorectal cancer?
How does the tumor microenvironment promote immunotherapy resistance?
What are the latest findings on ctDNA as a biomarker for early cancer detection?
```

---

## 🏭 Industry Use Cases

| Sector | Application |
|---|---|
| **Pharma R&D** | Query mechanism-of-action literature, drug resistance pathways |
| **Clinical Genomics** | Variant classification evidence lookup, ACMG guideline queries |
| **Diagnostics** | Biomarker validation literature, assay performance benchmarks |
| **Drug Safety** | Adverse event signal mining, hepatotoxicity literature review |
| **Ag-Biotech** | Crop gene editing, disease resistance QTL research |

---

## ⚙️ Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model for generation |
| `EMBEDDING_MODEL` | `NeuML/pubmedbert-base-embeddings` | Domain-specific biomedical embeddings |
| `CHUNK_SIZE` | `512` | Token size per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `TOP_K` | `8` | Candidate chunks retrieved |
| `RERANK_TOP_K` | `4` | Final chunks after re-ranking |

---

## 🔑 Key Design Decisions

**PubMed-BERT over general embeddings**
Biomedical domain-specific embeddings outperform general models (e.g. OpenAI `text-embedding-ada`) by 15–25% on gene/disease/pathway similarity benchmarks (BEIR).

**Hybrid BM25 + dense retrieval**
Exact gene symbols like `EGFR`, `BRAF V600E`, or drug names are handled by BM25's keyword matching; semantic similarity covers paraphrased or conceptual queries. Neither alone is sufficient for biomedical text.

**Agentic live fetch**
For queries containing "recent", "latest", "2024/2025", the agent bypasses the static knowledge base and hits PubMed live, preventing stale answers in a fast-moving field.

**Citation-enforced prompting**
The system prompt mandates PMID and UniProt citation inline, making every answer auditable — critical for regulatory, clinical, and publication contexts.

**ChromaDB for persistence**
Vector store persists to disk (`./chroma_db`). Restart the server without re-ingesting — your indexed papers are retained between sessions.

---

## 🛠️ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `GroqError: api_key not set` | `.env` not loaded | Check `.env` is in project root with no spaces around `=` |
| `RepositoryNotFoundError` | Wrong HuggingFace model name | Use `NeuML/pubmedbert-base-embeddings` in `config.py` |
| `ValueError: metadata value... dict` | Nested PubMed XML fields | Update `vector_store.py` with `sanitize_metadata()` function |
| `500 Internal Server Error` on ingest | See terminal logs | Check terminal running uvicorn for full traceback |
| Frontend shows "offline" | CORS or server not running | Add CORS middleware; ensure uvicorn is running on port 8000 |
| `ModuleNotFoundError` | Wrong working directory | Run commands from inside `bio-rag/` folder |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | REST API server |
| `chromadb` | Persistent vector store |
| `sentence-transformers` | PubMed-BERT embeddings + cross-encoder re-ranking |
| `groq` | LLM inference (LLaMA 3) |
| `biopython` | NCBI/Entrez interface |
| `rank-bm25` | Sparse keyword retrieval |
| `pypdf` | Local PDF ingestion |
| `xmltodict` | PubMed XML parsing |
| `langchain-community` | Document utilities |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [PubMed / NCBI](https://pubmed.ncbi.nlm.nih.gov/) for the biomedical literature API
- [UniProt](https://www.uniprot.org/) for protein function annotations
- [KEGG](https://www.genome.jp/kegg/) for biological pathway data
- [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) for domain-specific embeddings
- [Groq](https://groq.com/) for fast LLM inference
