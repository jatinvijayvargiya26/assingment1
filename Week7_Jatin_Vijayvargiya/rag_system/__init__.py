"""
RAG System Package
Document Question Answering using Retrieval-Augmented Generation
"""

from .pipeline import RAGPipeline
from .ingestion import DocumentIngestion
from .chunking import TextChunker
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import AnswerGenerator

__all__ = [
    "RAGPipeline",
    "DocumentIngestion",
    "TextChunker",
    "EmbeddingModel",
    "VectorStore",
    "Retriever",
    "AnswerGenerator",
]

__version__ = "1.0.0"
