"""
Embedding Model Module
======================
Step 3 of the RAG Pipeline.

Converts text chunks into dense vector representations using a
pre-trained sentence transformer model.

Model: all-MiniLM-L6-v2  (22MB, fast, high quality for semantic search)
- 384-dimensional output vectors
- Trained on 1 billion sentence pairs
- Great balance of speed and accuracy

Author: Week 7 Assignment
"""

import numpy as np
from typing import List, Union
from tqdm import tqdm


class EmbeddingModel:
    """
    Wrapper around a HuggingFace sentence-transformer embedding model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: 'all-MiniLM-L6-v2'
        Other options tried in this project:
        - 'all-mpnet-base-v2'  (higher quality, slower)
        - 'paraphrase-MiniLM-L3-v2' (fastest, lower quality)
    batch_size : int
        Number of texts to embed per batch (for memory efficiency).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # Lazy load on first use
        print(f"[Embeddings] Model set to '{model_name}'. Will load on first use.")

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            print(f"[Embeddings] Loading '{self.model_name}' (this may take a moment on first run)...")
            self._model = SentenceTransformer(self.model_name)
            try:
                dim = self._model.get_embedding_dimension()
            except AttributeError:
                dim = self._model.get_sentence_embedding_dimension()
            print(f"[Embeddings] Model loaded. Vector dimension = {dim}.")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load_model()
        try:
            return self._model.get_embedding_dimension()
        except AttributeError:
            return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Convert a list of text strings into a numpy matrix of embeddings.

        Parameters
        ----------
        texts : List[str]
            Input texts to embed.
        show_progress : bool
            Show a tqdm progress bar when embedding many texts.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim) — dtype float32
        """
        self._load_model()
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        print(f"[Embeddings] Embedding {len(texts)} texts in batches of {self.batch_size}...")

        all_embeddings = []
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        iterator = tqdm(batches, desc="  Encoding batches", unit="batch") if show_progress else batches
        for batch in iterator:
            batch_embeddings = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2-normalize for cosine similarity
                show_progress_bar=False,
            )
            all_embeddings.append(batch_embeddings)

        result = np.vstack(all_embeddings).astype(np.float32)
        print(f"[Embeddings] Done. Output shape: {result.shape}")
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray of shape (1, embedding_dim) — ready for FAISS search
        """
        self._load_model()
        vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return vector

    def get_config(self) -> dict:
        """Return embedding configuration for logging/metrics."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.dimension,
            "batch_size": self.batch_size,
            "normalization": "L2 (cosine similarity ready)",
        }
