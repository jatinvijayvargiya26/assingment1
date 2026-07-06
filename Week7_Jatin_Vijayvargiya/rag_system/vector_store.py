"""
Vector Store Module
===================
Step 4 of the RAG Pipeline.

Stores chunk embeddings in a FAISS (Facebook AI Similarity Search) index
for fast approximate nearest-neighbor (ANN) retrieval.

Why FAISS?
- Runs entirely in-memory, no server needed
- Scales to millions of vectors on a single machine
- Supports multiple index types; we use IndexFlatIP for exact search
  (inner product, equivalent to cosine similarity with L2-normalized vectors)

Author: Week 7 Assignment
"""

import os
import json
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional


class VectorStore:
    """
    FAISS-backed vector database for storing and searching chunk embeddings.

    Parameters
    ----------
    dimension : int
        Embedding vector dimensionality (must match the embedding model).
    index_type : str
        'flat_ip'  — exact inner product search (cosine similarity if normalized)
        'flat_l2'  — exact L2 distance search
    """

    def __init__(self, dimension: int, index_type: str = "flat_ip"):
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type
        self._index = self._build_index()
        self._chunks: List = []      # Parallel list: chunk objects per vector
        self._id_map: Dict[int, int] = {}   # vector_id → chunk list index
        print(f"[VectorStore] Initialized. Type={index_type}, Dimension={dimension}.")

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self):
        """Create and return a FAISS index based on index_type."""
        if self.index_type == "flat_ip":
            return self._faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "flat_l2":
            return self._faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unknown index_type '{self.index_type}'. Use 'flat_ip' or 'flat_l2'.")

    # ── Core operations ───────────────────────────────────────────────────────

    def add_chunks(self, chunks: List, embeddings: np.ndarray):
        """
        Add chunk objects and their corresponding embedding vectors to the store.

        Parameters
        ----------
        chunks : List[Chunk]
            Chunk objects (from chunking.py)
        embeddings : np.ndarray
            Shape (n_chunks, dimension), dtype float32
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )
        embeddings = embeddings.astype(np.float32)
        start_id = len(self._chunks)
        self._index.add(embeddings)
        for i, chunk in enumerate(chunks):
            self._id_map[start_id + i] = start_id + i
            self._chunks.append(chunk)
        print(f"[VectorStore] Added {len(chunks)} vectors. Total stored: {self._index.ntotal}.")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple]:
        """
        Search the index for the top-k most similar vectors.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape (1, dimension) — the embedded query.
        top_k : int
            Number of results to return.

        Returns
        -------
        List of (score, Chunk) tuples, sorted by score descending.
        """
        if self._index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Add chunks before searching.")
        top_k = min(top_k, self._index.ntotal)
        query_vector = query_vector.astype(np.float32)
        scores, indices = self._index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._chunks):
                results.append((float(score), self._chunks[idx]))
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: str):
        """Save the FAISS index and chunk metadata to disk."""
        os.makedirs(save_dir, exist_ok=True)
        index_path = os.path.join(save_dir, "faiss.index")
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        config_path = os.path.join(save_dir, "config.json")

        self._faiss.write_index(self._index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": self._index.ntotal,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[VectorStore] Saved index ({self._index.ntotal} vectors) to '{save_dir}'.")

    def load(self, save_dir: str):
        """Load a previously saved FAISS index and chunk metadata from disk."""
        index_path = os.path.join(save_dir, "faiss.index")
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        config_path = os.path.join(save_dir, "config.json")

        if not all(os.path.exists(p) for p in [index_path, chunks_path, config_path]):
            raise FileNotFoundError(f"Incomplete or missing vector store in '{save_dir}'.")

        with open(config_path) as f:
            config = json.load(f)
        self.dimension = config["dimension"]
        self.index_type = config["index_type"]
        self._index = self._faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)
        print(f"[VectorStore] Loaded index with {self._index.ntotal} vectors from '{save_dir}'.")

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return vector store statistics for the metrics report."""
        return {
            "total_vectors": self._index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "index_description": "FAISS IndexFlatIP — exact cosine similarity (L2-normalized vectors)",
        }
