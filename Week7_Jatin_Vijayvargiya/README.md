# Week 7: Document Question Answering System (RAG)

## Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline that answers questions from custom documents. Instead of relying on a language model's internal knowledge alone, the system retrieves relevant information from documents and generates answers grounded in that information.

## Project Structure

```
week7/
├── rag_system/
│   ├── __init__.py         # Package exports
│   ├── ingestion.py        # Step 1: Document ingestion (PDF, TXT, HuggingFace)
│   ├── chunking.py         # Step 2: Text chunking (recursive, fixed, sentence)
│   ├── embeddings.py       # Step 3: Sentence-transformer embeddings
│   ├── vector_store.py     # Step 4: FAISS vector database
│   ├── retriever.py        # Step 5 & 6: Query embedding + retrieval + hybrid search
│   ├── generator.py        # Step 7: LLM answer generation (Flan-T5)
│   └── pipeline.py         # End-to-end orchestrator
├── sample_docs/
│   └── sample.txt          # Built-in AI/ML reference document for testing
├── main.py                 # Entry point (demo + interactive mode)
├── validation_log.py       # Validation logs + system metrics report
├── requirements.txt        # All dependencies
└── README.md
```

## Setup & Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** On the first run, sentence-transformers (~90MB) and Flan-T5-base (~250MB) will be downloaded automatically from HuggingFace. Internet access is required only on the first run.

## How to Run

### Demo Mode (uses built-in sample document)
```bash
python main.py
```

### Use your own TXT file
```bash
python main.py --txt path/to/your_document.txt
```

### Use your own PDF
```bash
python main.py --pdf path/to/your_document.pdf
```

### Use a HuggingFace dataset
```bash
python main.py --hf squad --hf-samples 300
```

### Interactive Q&A mode (type your own questions)
```bash
python main.py --interactive
python main.py --txt my_notes.txt --interactive
```

### Enable Hybrid Search (Step 8 optimization)
```bash
python main.py --hybrid
python main.py --txt notes.txt --hybrid --interactive
```

### Save FAISS index to disk (avoid re-embedding next time)
```bash
python main.py --save-index ./saved_index
```

## Pipeline Architecture

```
Documents (PDF / TXT / HuggingFace)
        │
        ▼ Step 1: Document Ingestion
   Raw Text Documents
        │
        ▼ Step 2: Text Chunking (recursive, 512 chars, 128 overlap)
   Text Chunks  [Chunk 1] [Chunk 2] ... [Chunk N]
        │
        ▼ Step 3: Embedding Model (all-MiniLM-L6-v2, 384-dim)
   Embeddings  [v1] [v2] ... [vN]
        │
        ▼ Step 4: FAISS Vector Store (IndexFlatIP, cosine similarity)
   Vector Index
        │
   User Query ──▶ Step 5: Query Embedding (same model)
                       │
                       ▼ Step 6: Similarity Search → Top-K Chunks
                  Retrieved Context
                       │
                       ▼ Step 7: Prompt Construction + LLM (Flan-T5-base)
                  Grounded Answer
```

## Components & Technologies

| Component | Technology | Reason |
|-----------|-----------|--------|
| Document loading | PyMuPDF (PDF), built-in (TXT), `datasets` (HuggingFace) | Multi-format support |
| Text chunking | Custom recursive splitter | Boundary-aware, configurable overlap |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | Fast, 384-dim, high quality |
| Vector database | FAISS `IndexFlatIP` | Free, local, exact cosine search |
| Language model | `google/flan-t5-base` | Free, runs locally, seq2seq |
| Hybrid search | BM25 (rank-bm25) + vector | Best of keyword + semantic search |

## Step 8 Optimizations Implemented

1. **Hybrid Search**: Combines BM25 keyword scores with vector similarity scores using a weighted alpha parameter (`--hybrid` flag). This improves recall for queries that rely on specific keywords.

2. **Chunk overlap**: Consecutive chunks share 128 characters of context to prevent important information from being split at chunk boundaries.

3. **L2-normalized embeddings**: All vectors are L2-normalized before storage, making inner product equivalent to cosine similarity — ideal for semantic search.

4. **Batch embedding**: Chunks are embedded in batches of 32 to optimize GPU/CPU memory usage.

5. **Score threshold filtering**: Optionally discard retrieved chunks below a minimum similarity threshold.

## Output Files

- `validation_log.json` — Structured log of all query-answer pairs with retrieval scores and latency
- `saved_index/` (optional) — Persisted FAISS index for quick reload

## Sample Questions to Try

- "What is Retrieval-Augmented Generation and how does it work?"
- "What are the advantages of RAG over purely generative models?"
- "Explain the difference between supervised and unsupervised learning."
- "What is FAISS and why is it used?"
- "What is the T5 model architecture?"
- "What are the stages of the RAG pipeline?"
- "How does hybrid search work?"

## Key Learnings

- How RAG systems combine retrieval and generation for grounded answers
- The role of vector embeddings in semantic similarity search
- How FAISS enables fast similarity search over large document collections
- Chunking strategies and why overlap matters
- How to integrate local open-source LLMs for free inference
- The importance of retrieval quality for generation accuracy
