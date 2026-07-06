"""
Document Ingestion Module
========================
Step 1 of the RAG Pipeline.

Supports:
- Raw .txt files
- PDF documents (via PyMuPDF)
- HuggingFace dataset archives
- Plain string input

Author: Week 7 Assignment
"""

import os
import re
from typing import List, Dict, Optional

# ── PDF support ──────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[WARNING] PyMuPDF not installed. PDF ingestion disabled. Install: pip install PyMuPDF")

# ── HuggingFace datasets support ─────────────────────────────────────────────
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[WARNING] 'datasets' library not installed. HuggingFace ingestion disabled.")


class Document:
    """Represents a single ingested document with text and metadata."""

    def __init__(self, text: str, source: str, page: Optional[int] = None, metadata: Optional[Dict] = None):
        self.text = text.strip()
        self.source = source
        self.page = page
        self.metadata = metadata or {}

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Document(source='{self.source}', page={self.page}, preview='{preview}...')"


class DocumentIngestion:
    """
    Document Ingestion Module.

    Handles loading documents from multiple sources and returns a
    unified list of Document objects for downstream chunking and embedding.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.loaded_documents: List[Document] = []

    def _log(self, message: str):
        if self.verbose:
            print(f"[Ingestion] {message}")

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, source_name: str = "manual_input") -> List[Document]:
        """Ingest a raw text string directly."""
        if not text.strip():
            raise ValueError("Input text is empty.")
        doc = Document(text=text, source=source_name)
        self._log(f"Ingested raw text ({len(text)} characters) from '{source_name}'.")
        self.loaded_documents.append(doc)
        return [doc]

    def ingest_txt_file(self, filepath: str) -> List[Document]:
        """Load and ingest a plain .txt file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        doc = Document(text=text, source=filepath)
        self._log(f"Loaded TXT file: '{filepath}' ({len(text)} chars).")
        self.loaded_documents.append(doc)
        return [doc]

    def ingest_pdf(self, filepath: str) -> List[Document]:
        """
        Load a PDF file and extract text page-by-page using PyMuPDF.
        Each page becomes a separate Document for fine-grained retrieval.
        """
        if not PDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF ingestion. Run: pip install PyMuPDF")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF not found: {filepath}")

        documents = []
        pdf = fitz.open(filepath)
        self._log(f"Loading PDF: '{filepath}' — {len(pdf)} pages detected.")
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")
            # Skip nearly-empty pages (headers/footers only)
            if len(text.strip()) < 30:
                continue
            doc = Document(text=text, source=filepath, page=page_num + 1)
            documents.append(doc)
        pdf.close()
        self._log(f"Extracted text from {len(documents)} pages of '{os.path.basename(filepath)}'.")
        self.loaded_documents.extend(documents)
        return documents

    def ingest_folder(self, folder_path: str, extensions: List[str] = None) -> List[Document]:
        """
        Recursively load all documents from a directory.
        Supported extensions by default: ['.txt', '.pdf']
        """
        if extensions is None:
            extensions = [".txt", ".pdf"]
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        all_docs = []
        for root, _, files in os.walk(folder_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(root, fname)
                    try:
                        if ext == ".txt":
                            all_docs.extend(self.ingest_txt_file(full_path))
                        elif ext == ".pdf":
                            all_docs.extend(self.ingest_pdf(full_path))
                    except Exception as e:
                        self._log(f"  [SKIP] Could not load '{fname}': {e}")
        self._log(f"Folder ingestion complete. Total docs loaded: {len(all_docs)}")
        return all_docs

    def ingest_huggingface_dataset(
        self,
        dataset_name: str = "squad",
        split: str = "train",
        text_column: str = "context",
        max_samples: int = 100,
    ) -> List[Document]:
        """
        Ingest text from a HuggingFace dataset.
        Default: SQuAD dataset (context column) — good beginner dataset.
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Install 'datasets': pip install datasets")

        self._log(f"Loading HuggingFace dataset: '{dataset_name}' (split='{split}', max={max_samples})...")
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        # Deduplicate by context text
        seen_texts = set()
        documents = []
        for idx, row in enumerate(dataset):
            if idx >= max_samples:
                break
            text = row.get(text_column, "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                doc = Document(
                    text=text,
                    source=f"{dataset_name}/{split}",
                    metadata={"dataset": dataset_name, "split": split, "index": idx},
                )
                documents.append(doc)
        self._log(f"Loaded {len(documents)} unique passages from '{dataset_name}'.")
        self.loaded_documents.extend(documents)
        return documents

    def get_all_documents(self) -> List[Document]:
        """Return all documents ingested so far."""
        return self.loaded_documents

    def get_stats(self) -> Dict:
        """Return ingestion statistics."""
        total_chars = sum(len(d.text) for d in self.loaded_documents)
        sources = list({d.source for d in self.loaded_documents})
        return {
            "total_documents": len(self.loaded_documents),
            "total_characters": total_chars,
            "unique_sources": sources,
        }
