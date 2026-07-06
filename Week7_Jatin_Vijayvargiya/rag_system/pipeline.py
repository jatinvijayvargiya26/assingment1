"""
RAG Pipeline — End-to-End Orchestrator
=======================================
Connects all 7 pipeline stages into a single, easy-to-use class.

Pipeline Stages:
  1. Document Ingestion   (ingestion.py)
  2. Text Chunking        (chunking.py)
  3. Embedding Creation   (embeddings.py)
  4. Vector Store         (vector_store.py)
  5. Query Embedding      (retriever.py)
  6. Context Retrieval    (retriever.py)
  7. Answer Generation    (generator.py)
  8. Optimizations        (hybrid search, re-ranking in retriever.py)

Author: Week 7 Assignment
"""

import os
import time
from typing import List, Optional, Dict

from .ingestion import DocumentIngestion, Document
from .chunking import TextChunker, Chunk
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import AnswerGenerator


class RAGPipeline:
    """
    Full Retrieval-Augmented Generation Pipeline.

    Usage
    -----
    >>> pipeline = RAGPipeline()
    >>> pipeline.ingest_txt("my_document.txt")
    >>> pipeline.build_index()
    >>> result = pipeline.query("What is the main topic of the document?")
    >>> print(result['answer'])
    """

    def __init__(
        self,
        # Chunking config
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        chunk_strategy: str = "recursive",
        # Embedding config
        embedding_model_name: str = "all-MiniLM-L6-v2",
        # Retrieval config
        top_k: int = 5,
        score_threshold: float = 0.0,
        use_hybrid_search: bool = False,
        hybrid_alpha: float = 0.7,
        # Generation config
        generator_model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 256,
        # Save/load directory
        index_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.index_dir = index_dir
        self._is_indexed = False
        self._all_chunks: List[Chunk] = []

        print("\n" + "=" * 60)
        print("  RAG Pipeline — Document Question Answering System")
        print("=" * 60)

        # Initialize all components
        self.ingestion = DocumentIngestion(verbose=verbose)
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy,
        )
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name,
            batch_size=32,
        )
        # Vector store initialized after first embedding (need dimension)
        self._vector_store = None
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._use_hybrid = use_hybrid_search
        self._hybrid_alpha = hybrid_alpha
        self._generator_model = generator_model_name
        self._max_new_tokens = max_new_tokens
        self.retriever = None
        self.generator = AnswerGenerator(
            model_name=generator_model_name,
            max_new_tokens=max_new_tokens,
        )

    # ── Ingestion helpers ─────────────────────────────────────────────────────

    def ingest_txt(self, filepath: str):
        """Add a .txt file to the pipeline."""
        self.ingestion.ingest_txt_file(filepath)
        self._is_indexed = False

    def ingest_pdf(self, filepath: str):
        """Add a PDF file to the pipeline."""
        self.ingestion.ingest_pdf(filepath)
        self._is_indexed = False

    def ingest_text(self, text: str, source_name: str = "custom_input"):
        """Add a raw text string to the pipeline."""
        self.ingestion.ingest_text(text, source_name=source_name)
        self._is_indexed = False

    def ingest_huggingface(self, dataset_name: str = "squad", split: str = "train",
                           text_column: str = "context", max_samples: int = 100):
        """Add text from a HuggingFace dataset."""
        self.ingestion.ingest_huggingface_dataset(
            dataset_name=dataset_name,
            split=split,
            text_column=text_column,
            max_samples=max_samples,
        )
        self._is_indexed = False

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self):
        """
        Run Stages 2–4: Chunk documents → embed → store in FAISS.
        Must be called before any queries.
        """
        documents = self.ingestion.get_all_documents()
        if not documents:
            raise RuntimeError("No documents ingested. Call ingest_txt(), ingest_pdf(), etc. first.")

        print(f"\n[Pipeline] Starting index build with {len(documents)} document(s)...")
        t_start = time.time()

        # Stage 2: Chunk
        print("\n[Pipeline] Stage 2: Text Chunking")
        self._all_chunks = self.chunker.chunk_documents(documents)
        if not self._all_chunks:
            raise RuntimeError("No chunks produced. Documents may be empty.")

        # Stage 3: Embed
        print("\n[Pipeline] Stage 3: Creating Embeddings")
        chunk_texts = [c.text for c in self._all_chunks]
        embeddings = self.embedding_model.embed_texts(chunk_texts)

        # Stage 4: Vector store
        print("\n[Pipeline] Stage 4: Building Vector Store")
        dim = embeddings.shape[1]
        self._vector_store = VectorStore(dimension=dim, index_type="flat_ip")
        self._vector_store.add_chunks(self._all_chunks, embeddings)

        # Initialize retriever
        self.retriever = Retriever(
            vector_store=self._vector_store,
            embedding_model=self.embedding_model,
            top_k=self._top_k,
            score_threshold=self._score_threshold,
            use_hybrid_search=self._use_hybrid,
            hybrid_alpha=self._hybrid_alpha,
        )
        if self._use_hybrid:
            self.retriever.build_bm25_index(self._all_chunks)

        # Save index if requested
        if self.index_dir:
            self._vector_store.save(self.index_dir)

        elapsed = time.time() - t_start
        self._is_indexed = True
        print(f"\n[Pipeline] Index built in {elapsed:.1f}s. Ready for queries!")
        print("=" * 60)

    def load_index(self, load_dir: str):
        """Load a pre-built vector index from disk (skip re-embedding)."""
        self._vector_store = VectorStore(dimension=384)  # will be overwritten by load
        self._vector_store.load(load_dir)
        self.retriever = Retriever(
            vector_store=self._vector_store,
            embedding_model=self.embedding_model,
            top_k=self._top_k,
            score_threshold=self._score_threshold,
        )
        self._is_indexed = True
        print(f"[Pipeline] Index loaded from '{load_dir}'.")

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, show_sources: bool = True) -> Dict:
        """
        Run a complete RAG query: embed → retrieve → generate.

        Parameters
        ----------
        question : str
            Natural language question.
        show_sources : bool
            If True, print source attribution.

        Returns
        -------
        dict with: 'answer', 'sources', 'scores', 'retrieved_chunks', 'prompt'
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        print(f"\n{'─' * 60}")
        print(f"  QUERY: {question}")
        print(f"{'─' * 60}")

        # Stages 5 & 6: Retrieve
        retrieved = self.retriever.retrieve(question)

        # Stage 7: Generate
        result = self.generator.generate(question, retrieved)
        result["retrieved_chunks"] = retrieved

        # Print answer
        print(f"\n  ANSWER:\n  {result['answer']}")

        if show_sources and result["sources"]:
            print(f"\n  SOURCES:")
            for i, (src, score) in enumerate(zip(result["sources"], result["scores"]), 1):
                print(f"    {i}. {src}  [similarity: {score:.4f}]")
        print(f"{'─' * 60}")
        return result

    # ── Metrics & Stats ───────────────────────────────────────────────────────

    def get_system_report(self) -> Dict:
        """
        Generate a full system metrics report covering all pipeline components.
        Fulfills the assignment requirement for system metrics documentation.
        """
        report = {
            "ingestion_stats": self.ingestion.get_stats(),
            "chunking_config": self.chunker.get_config(),
            "embedding_config": self.embedding_model.get_config(),
            "vector_store_stats": self._vector_store.get_stats() if self._vector_store else "Not built",
            "retrieval_stats": self.retriever.get_stats() if self.retriever else "Not initialized",
            "generator_config": self.generator.get_config(),
        }
        return report
