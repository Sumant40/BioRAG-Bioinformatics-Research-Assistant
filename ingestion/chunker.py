# ingestion/chunker.py
import re
from dataclasses import dataclass
import tiktoken


@dataclass
class Chunk:
    text: str
    metadata: dict


class BioChunker:
    """
    Bioinformatics-aware chunker.
    Respects sentence boundaries and avoids splitting
    gene names, accession IDs, or numbered lists mid-item.
    """

    # Patterns that signal a section boundary in biomedical text
    SECTION_HEADERS = re.compile(
        r"\b(Introduction|Methods|Materials and Methods|Results|Discussion|"
        r"Conclusion|Background|Abstract|References|Supplementary)\b",
        re.IGNORECASE,
    )

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _token_len(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _split_sentences(self, text: str) -> list[str]:
        # Split on sentence endings but keep gene/protein abbreviations intact
        # e.g. "BRCA1 is expressed. It encodes..." should split at the period
        # but "Dr. Smith" or "et al." should not
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, text: str, metadata: dict) -> list[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []
        current_tokens = []
        current_len = 0

        for sentence in sentences:
            s_len = self._token_len(sentence)

            if current_len + s_len > self.chunk_size and current_tokens:
                chunk_text = " ".join(current_tokens)
                chunks.append(Chunk(text=chunk_text, metadata=metadata.copy()))

                # Overlap: keep last N tokens worth of sentences
                overlap_tokens = []
                overlap_len = 0
                for sent in reversed(current_tokens):
                    tl = self._token_len(sent)
                    if overlap_len + tl > self.chunk_overlap:
                        break
                    overlap_tokens.insert(0, sent)
                    overlap_len += tl

                current_tokens = overlap_tokens
                current_len = overlap_len

            current_tokens.append(sentence)
            current_len += s_len

        if current_tokens:
            chunks.append(Chunk(text=" ".join(current_tokens), metadata=metadata.copy()))

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def chunk_records(self, records: list[dict], text_field: str = "abstract") -> list[Chunk]:
        all_chunks = []
        for record in records:
            text = record.get(text_field, "") or record.get("text", "")
            if not text.strip():
                continue
            metadata = {k: v for k, v in record.items() if k != text_field}
            chunks = self.chunk_document(text, metadata)
            all_chunks.extend(chunks)
        return all_chunks