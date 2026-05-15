# agents/bio_agent.py
import re
from agents.tools import BioTools


GENE_PATTERN = re.compile(r'\b[A-Z][A-Z0-9]{1,7}\b')   # rough HGNC symbol pattern
PATHWAY_PATTERN = re.compile(r'\bhsa\d{5}\b')            # KEGG human pathway IDs


class BioAgent:
    """
    Agentic router that decides whether to:
    1. Answer from the local vector store (fast, curated)
    2. Trigger a live PubMed search (recent events, niche queries)
    3. Look up a specific protein on UniProt (gene-centric queries)
    4. Fetch a KEGG pathway (pathway analysis queries)
    """

    def __init__(self, retriever, reranker, generator, config):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.tools = BioTools(config)

    def _classify_query(self, query: str) -> dict:
        """Simple rule-based intent classification.
        In production, replace with an LLM call for intent detection.
        """
        q_lower = query.lower()
        intent = {
            "needs_live_pubmed": any(
                kw in q_lower for kw in ["recent", "latest", "2024", "2025", "new study"]
            ),
            "gene_lookup": bool(GENE_PATTERN.search(query)),
            "pathway_query": bool(PATHWAY_PATTERN.search(query))
                             or "pathway" in q_lower,
        }
        return intent

    def run(self, query: str) -> dict:
        intent = self._classify_query(query)
        extra_context = []

        # Tool: live PubMed fetch
        if intent["needs_live_pubmed"]:
            live_records = self.tools.search_pubmed_live(query, max_results=5)
            for r in live_records:
                extra_context.append({
                    "text": f"{r['title']}. {r['abstract']}",
                    "metadata": r,
                    "score": 0.7,     # synthetic score for live results
                    "hybrid_score": 0.7,
                })

        # Tool: UniProt gene lookup
        if intent["gene_lookup"]:
            genes = GENE_PATTERN.findall(query)
            for gene in genes[:2]:    # cap at 2 genes per query
                protein_records = self.tools.lookup_protein(gene)
                for pr in protein_records:
                    text = (f"Gene: {pr['gene']} | Protein: {pr['protein_name']}\n"
                            f"Function: {pr['function']}\n"
                            f"Disease association: {pr['disease_association']}")
                    extra_context.append({
                        "text": text,
                        "metadata": pr,
                        "score": 0.75,
                        "hybrid_score": 0.75,
                    })

        # Tool: KEGG pathway
        if intent["pathway_query"]:
            pathways = PATHWAY_PATTERN.findall(query)
            for pid in pathways[:1]:
                pdata = self.tools.query_kegg_pathway(pid)
                extra_context.append({
                    "text": pdata["data"],
                    "metadata": {"source": "kegg", "pathway_id": pid},
                    "score": 0.72,
                    "hybrid_score": 0.72,
                })

        # Always retrieve from vector store
        vector_hits = self.retriever.retrieve(query)
        all_hits = extra_context + vector_hits

        # Re-rank combined results
        reranked = self.reranker.rerank(query, all_hits)

        # Generate grounded answer
        result = self.generator.generate(query, reranked)
        result["intent"] = intent
        result["live_fetch"] = bool(extra_context)

        return result