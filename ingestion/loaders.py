# ingestion/loaders.py
import requests
import xmltodict
from pathlib import Path
from pypdf import PdfReader
from Bio import Entrez, SeqIO
import json

Entrez.email = "sumantjadiyappagoudar@gmail.com"  # required by NCBI


class PubMedLoader:
    """Fetch abstracts and full-text metadata from PubMed."""

    def __init__(self, config):
        self.base_url = config.PUBMED_BASE_URL
        self.api_key = config.NCBI_API_KEY

    def search(self, query: str, max_results: int = 50) -> list[dict]:
        """Search PubMed and return structured records."""
        # Step 1: get PMIDs
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "api_key": self.api_key,
        }
        r = requests.get(search_url, params=params)
        pmids = r.json()["esearchresult"]["idlist"]

        if not pmids:
            return []

        # Step 2: fetch full records
        fetch_url = f"{self.base_url}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "api_key": self.api_key,
        }
        r = requests.get(fetch_url, params=params)
        data = xmltodict.parse(r.content)

        articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        if isinstance(articles, dict):
            articles = [articles]

        records = []
        for article in articles:
            try:
                medline = article["MedlineCitation"]
                art = medline["Article"]
                abstract_text = art.get("Abstract", {}).get("AbstractText", "")

                # Abstract can be a list of structured sections
                if isinstance(abstract_text, list):
                    abstract = " ".join(
                        [a.get("#text", a) if isinstance(a, dict) else a
                         for a in abstract_text]
                    )
                elif isinstance(abstract_text, dict):
                    abstract = abstract_text.get("#text", "")
                else:
                    abstract = str(abstract_text)

                records.append({
                    "pmid": medline["PMID"]["#text"],
                    "title": art.get("ArticleTitle", ""),
                    "abstract": abstract,
                    "journal": art.get("Journal", {}).get("Title", ""),
                    "year": art.get("Journal", {}).get("JournalIssue", {})
                               .get("PubDate", {}).get("Year", ""),
                    "source": "pubmed",
                })
            except Exception:
                continue

        return records


class UniProtLoader:
    """Fetch protein function annotations from UniProt."""

    def __init__(self, config):
        self.base_url = config.UNIPROT_BASE_URL

    def fetch_protein(self, gene_name: str, organism: str = "homo sapiens") -> list[dict]:
        url = f"{self.base_url}/uniprotkb/search"
        params = {
            "query": f"gene:{gene_name} AND organism_name:{organism} AND reviewed:true",
            "format": "json",
            "size": 5,
            "fields": "accession,gene_names,protein_name,function,disease,go",
        }
        r = requests.get(url, params=params)
        results = r.json().get("results", [])

        records = []
        for entry in results:
            comments = entry.get("comments", [])
            function_text = ""
            disease_text = ""

            for c in comments:
                if c.get("commentType") == "FUNCTION":
                    texts = c.get("texts", [])
                    function_text = " ".join(t.get("value", "") for t in texts)
                if c.get("commentType") == "DISEASE":
                    disease = c.get("disease", {})
                    disease_text += disease.get("diseaseId", "") + ": "
                    disease_text += " ".join(
                        t.get("value", "") for t in c.get("texts", [])
                    )

            records.append({
                "accession": entry.get("primaryAccession", ""),
                "gene": gene_name,
                "protein_name": entry.get("proteinDescription", {})
                                     .get("recommendedName", {})
                                     .get("fullName", {}).get("value", ""),
                "function": function_text,
                "disease_association": disease_text,
                "source": "uniprot",
            })

        return records


class PDFLoader:
    """Load local PDF files — clinical guidelines, review papers."""

    def load(self, filepath: str) -> list[dict]:
        reader = PdfReader(filepath)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text,
                    "page": i + 1,
                    "source": Path(filepath).name,
                    "source_type": "pdf",
                })
        return pages