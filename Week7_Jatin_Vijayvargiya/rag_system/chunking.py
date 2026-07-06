"""
Text Chunking Module
====================
Step 2 of the RAG Pipeline.

Breaks raw document text into smaller, manageable chunks for embedding
and retrieval. Supports multiple chunking strategies:

1. Recursive character splitting (default) - smart boundary-aware splits
2. Fixed-size character chunking - simple but reliable
3. Sentence-level chunking - splits on sentence boundaries

Author: Week 7 Assignment
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single text chunk with positional metadata."""
    text: str
    source: str
    chunk_id: int
    start_char: int
    end_char: int
    page: Optional[int] = None
    metadata: Optional[Dict] = None

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, source='{self.source}', text='{preview}...')"


class TextChunker:
    """
    Text Chunking Module.

    Converts Document objects (or raw text strings) into overlapping
    Chunk objects for downstream embedding and retrieval.

    Parameters
    ----------
    chunk_size : int
        Target character length of each chunk (default 512).
    chunk_overlap : int
        Number of characters to overlap between consecutive chunks (default 128).
        Overlap prevents important context from being split at boundaries.
    strategy : str
        'recursive' | 'fixed' | 'sentence'
    """

    # Separators tried in order - paragraph, line, sentence, clause, word
    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, strategy: str = "recursive"):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self._chunk_counter = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_documents(self, documents) -> List[Chunk]:
        """
        Main entry point. Accepts a list of Document objects.
        Returns a flat list of Chunk objects across all documents.
        """
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc.text, source=doc.source, page=doc.page)
            all_chunks.extend(chunks)
        print(f"[Chunking] Strategy='{self.strategy}', size={self.chunk_size}, "
              f"overlap={self.chunk_overlap} -> {len(all_chunks)} chunks from {len(documents)} documents.")
        return all_chunks

    def chunk_text(self, text: str, source: str = "unknown", page: int = None) -> List[Chunk]:
        """Chunk a single raw text string. Convenience wrapper."""
        chunks = self._chunk_text(text, source, page)
        print(f"[Chunking] '{source}': {len(chunks)} chunks generated.")
        return chunks

    def get_config(self) -> Dict:
        """Return the current chunking configuration as a dict."""
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    # ── Internal logic ────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, source: str, page: Optional[int]) -> List[Chunk]:
        """Dispatch to the chosen chunking strategy."""
        text = self._clean_text(text)
        if not text:
            return []

        if self.strategy == "recursive":
            raw_chunks = self._recursive_split(text)
        elif self.strategy == "fixed":
            raw_chunks = self._fixed_split(text)
        elif self.strategy == "sentence":
            raw_chunks = self._sentence_split(text)
        else:
            raise ValueError(f"Unknown strategy: '{self.strategy}'. Use 'recursive', 'fixed', or 'sentence'.")

        # Build Chunk objects with character positions
        chunks = []
        cursor = 0
        for raw in raw_chunks:
            raw = raw.strip()
            if not raw:
                continue
            start = text.find(raw, cursor)
            if start == -1:
                start = cursor
            end = start + len(raw)
            chunk = Chunk(
                text=raw,
                source=source,
                chunk_id=self._chunk_counter,
                start_char=start,
                end_char=end,
                page=page,
                metadata={"strategy": self.strategy},
            )
            chunks.append(chunk)
            self._chunk_counter += 1
            cursor = max(0, end - self.chunk_overlap)
        return chunks

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and normalize line endings."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _recursive_split(self, text: str) -> List[str]:
        """
        Split text using a priority list of separators (non-recursive implementation).
        Tries each separator in priority order to find the best split point
        for each chunk. Falls back to fixed character splitting as last resort.
        """
        result_chunks = []
        remaining = text

        while len(remaining) > self.chunk_size:
            # Find best split point within [chunk_size - overlap, chunk_size]
            # Try each separator in priority order
            split_pos = -1
            for sep in self._SEPARATORS:
                # Look for separator near the end of the target chunk window
                # Search backwards from chunk_size to (chunk_size // 2) for a separator
                search_start = max(0, self.chunk_size // 2)
                search_end = self.chunk_size
                idx = remaining.rfind(sep, search_start, search_end)
                if idx != -1:
                    split_pos = idx + len(sep)
                    break

            if split_pos <= 0 or split_pos > len(remaining):
                # No good separator found — hard split at chunk_size
                split_pos = self.chunk_size

            chunk = remaining[:split_pos].strip()
            if chunk:
                result_chunks.append(chunk)

            # Advance with overlap
            next_start = max(0, split_pos - self.chunk_overlap)
            remaining = remaining[next_start:]

        # Add the remaining text as the final chunk
        if remaining.strip():
            result_chunks.append(remaining.strip())

        return result_chunks

    def _fixed_split(self, text: str) -> List[str]:
        """Simple fixed-size character splitting with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sentence_split(self, text: str) -> List[str]:
        """Split on sentence boundaries, then merge into chunks up to chunk_size."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.chunk_size and current:
                chunks.append(current.strip())
                # Overlap: start next chunk with last portion of current
                overlap = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = overlap + " " + sentence
            else:
                current = (current + " " + sentence).strip() if current else sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks
