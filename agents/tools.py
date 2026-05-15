# agents/tools.py
import requests
from ingestion.loaders import PubMedLoader, UniProtLoader


class BioTools:
    """
    Tools available to the bio-agent for live data fetching
    when the vector store doesn't have sufficient coverage.
    """

    def __init__(self, config):
        self.pubmed = PubMedLoader(config)
        self.uniprot = UniProtLoader(config)

    def search_pubmed_live(self, query: str, max_results: int = 5) -> list[dict]:
        """Live PubMed search — used when KB is stale or query is very specific."""
        return self.pubmed.search(query, max_results=max_results)

    def lookup_protein(self, gene_name: str) -> list[dict]:
        """Fetch protein function/disease data from UniProt."""
        return self.uniprot.fetch_protein(gene_name)

    def query_kegg_pathway(self, pathway_id: str) -> dict:
        """Fetch KEGG pathway info."""
        url = f"https://rest.kegg.jp/get/{pathway_id}"
        r = requests.get(url)
        return {"pathway_id": pathway_id, "data": r.text[:3000]}