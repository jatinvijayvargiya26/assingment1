"""
Retriever Module
================
Step 5 & 6 of the RAG Pipeline.

Implements:
- Step 5: Query input route (converts query string → embedding)
- Step 6: Retrieval — finds top-k relevant chunks from the vector store

Additionally supports:
- Hybrid Search (keyword BM25 + vector similarity) — Step 8 optimization
- Score threshold filtering (only return chunks above a confidence level)

Author: Week 7 Assignment
"""

import re
from typing import List, Tuple, Dict, Optional

from .vector_store import VectorStore
from .embeddings import EmbeddingModel
from .chunking import Chunk


class Retriever:
    """
    Retrieval Module.

    Converts a natural language query into an embedding and retrieves
    the most relevant document chunks from the vector store.

    Parameters
    ----------
    vector_store : VectorStore
        The initialized and populated FAISS vector store.
    embedding_model : EmbeddingModel
        The embedding model used to encode the query.
    top_k : int
        Number of top chunks to retrieve per query.
    score_threshold : float
        Minimum cosine similarity score to include a result (0.0 to 1.0).
        Set to 0.0 to disable filtering.
    use_hybrid_search : bool
        If True, combine BM25 keyword scores with vector similarity scores.
    hybrid_alpha : float
        Weight for vector score in hybrid search. (1-alpha) goes to BM25.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        score_threshold: float = 0.0,
        use_hybrid_search: bool = False,
        hybrid_alpha: float = 0.7,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_alpha = hybrid_alpha
        self._bm25 = None
        self._all_chunks: List[Chunk] = []
        self._query_count = 0
        print(f"[Retriever] Initialized. top_k={top_k}, hybrid={use_hybrid_search}, "
              f"threshold={score_threshold}")

    # ── BM25 Setup ────────────────────────────────────────────────────────────

    def build_bm25_index(self, chunks: List[Chunk]):
        """
        Build a BM25 keyword index from the provided chunks.
        Required if use_hybrid_search=True.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")

        self._all_chunks = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        print(f"[Retriever] BM25 index built on {len(chunks)} chunks.")

    # ── Query Input Route (Step 5) ────────────────────────────────────────────

    def embed_query(self, query: str):
        """
        Step 5: Convert an incoming user query into a vector representation.
        This is the query input route.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        query_vector = self.embedding_model.embed_query(query)
        print(f"[Retriever] Query embedded → vector shape {query_vector.shape}.")
        return query_vector

    # ── Retrieval (Step 6) ────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Tuple[float, Chunk]]:
        """
        Full retrieval pipeline for a given query.

        1. Embeds the query (Step 5)
        2. Searches vector store for top-k results
        3. Optionally applies BM25 re-ranking (Step 8 optimization)
        4. Filters by score threshold

        Returns
        -------
        List of (score, Chunk) tuples, sorted by relevance (highest first).
        """
        self._query_count += 1
        print(f"\n[Retriever] Processing query #{self._query_count}: '{query[:80]}...' " 
              if len(query) > 80 else f"\n[Retriever] Processing query #{self._query_count}: '{query}'")

        # Step 5: Embed query
        query_vector = self.embed_query(query)

        if self.use_hybrid_search and self._bm25 is not None:
            results = self._hybrid_retrieve(query, query_vector)
        else:
            # Step 6: Pure vector similarity search
            results = self.vector_store.search(query_vector, top_k=self.top_k)

        # Apply score threshold filter
        if self.score_threshold > 0.0:
            results = [(s, c) for s, c in results if s >= self.score_threshold]

        print(f"[Retriever] Retrieved {len(results)} relevant chunks.")
        return results

    def _hybrid_retrieve(self, query: str, query_vector) -> List[Tuple[float, Chunk]]:
        """
        Step 8 optimization: Hybrid retrieval combining BM25 and vector scores.

        Formula: hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
        Both scores are min-max normalized before combining.
        """
        # 1. Get vector scores (search more candidates for re-ranking)
        candidate_k = min(self.top_k * 3, self.vector_store._index.ntotal)
        vector_results = self.vector_store.search(query_vector, top_k=candidate_k)

        # 2. Get BM25 keyword scores for all chunks
        tokens = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(tokens)

        # 3. Build score lookup: chunk_id → vector_score
        vec_score_map = {}
        for score, chunk in vector_results:
            vec_score_map[chunk.chunk_id] = score

        # 4. Normalize both score arrays to [0, 1]
        vec_scores_arr = [vec_score_map.get(c.chunk_id, 0.0) for c in self._all_chunks]
        vec_norm = self._normalize(vec_scores_arr)
        bm25_norm = self._normalize(bm25_scores.tolist())

        # 5. Combine scores
        combined = []
        for i, chunk in enumerate(self._all_chunks):
            hybrid = self.hybrid_alpha * vec_norm[i] + (1 - self.hybrid_alpha) * bm25_norm[i]
            combined.append((hybrid, chunk))

        # 6. Sort and return top_k
        combined.sort(key=lambda x: x[0], reverse=True)
        print(f"[Retriever] Hybrid search: combined {len(combined)} candidates → top {self.top_k}.")
        return combined[: self.top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer for BM25."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        """Min-max normalize a list of scores to [0, 1]."""
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def get_stats(self) -> Dict:
        """Return retrieval statistics."""
        return {
            "queries_processed": self._query_count,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "hybrid_search_enabled": self.use_hybrid_search,
            "hybrid_alpha": self.hybrid_alpha if self.use_hybrid_search else "N/A",
        }
