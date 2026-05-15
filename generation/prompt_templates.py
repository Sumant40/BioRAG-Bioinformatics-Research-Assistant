# generation/prompt_templates.py

BIOMEDICAL_SYSTEM_PROMPT = """You are a specialist biomedical research assistant with
expertise in genomics, molecular biology, clinical genetics, and pharmacology.

Guidelines:
- Ground every claim in the provided context. Do not hallucinate gene functions,
  drug interactions, or clinical outcomes.
- Cite sources using [PMID: XXXXX] or [UniProt: ACCESSION] notation inline.
- Distinguish between established findings and preliminary/in-vitro evidence.
- Use standard nomenclature: HGNC gene symbols, OMIM IDs for diseases,
  IUPAC names or INN for compounds.
- When evidence conflicts, present both perspectives.
- If the context does not contain enough information, say so clearly
  rather than speculating.
"""

QUERY_TEMPLATE = """Context retrieved from biomedical databases and literature:

{context}

---

Research question: {query}

Provide a precise, evidence-based answer. Cite the source for each key claim.
"""

def build_context_block(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        source_tag = _format_source_tag(meta)
        blocks.append(f"[{i}] {source_tag}\n{hit['text']}")
    return "\n\n".join(blocks)

def _format_source_tag(meta: dict) -> str:
    if meta.get("source") == "pubmed":
        return (f"PubMed PMID:{meta.get('pmid','')} | "
                f"{meta.get('journal','')} {meta.get('year','')}")
    if meta.get("source") == "uniprot":
        return (f"UniProt:{meta.get('accession','')} | "
                f"Gene: {meta.get('gene','')}")
    return f"Source: {meta.get('source', 'local')}"