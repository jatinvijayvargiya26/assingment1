"""
Validation & Logging Module
============================
Generates documented validation logs and system metrics report.

Fulfills assignment requirements:
 "Documented validation logs showing accurate text extraction retrieval
  performance when tested with dynamic sample questions."
 "System metrics report detailing chunking profiles, chosen text embedding
  dimensions, vector store tools, and language model setups."

Author: Week 7 Assignment
"""

import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
from tabulate import tabulate


class ValidationLogger:
    """
    Logs query-answer pairs with retrieval metadata for performance validation.
    Writes structured logs to a JSON file for review.
    """

    def __init__(self, log_path: str = "validation_log.json"):
        self.log_path = log_path
        self.entries: List[Dict] = []
        print(f"[ValidationLogger] Logging to '{log_path}'.")

    def log_query(
        self,
        query: str,
        answer: str,
        retrieved_chunks,
        scores: List[float],
        sources: List[str],
        latency_sec: float,
    ):
        """Log a single query-answer interaction."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "latency_seconds": round(latency_sec, 3),
            "retrieved_count": len(retrieved_chunks),
            "top_score": round(max(scores), 4) if scores else 0.0,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "sources": sources,
            "retrieved_previews": [
                c.text[:120].replace("\n", " ") for _, c in retrieved_chunks
            ],
        }
        self.entries.append(entry)
        print(f"[ValidationLogger] Logged query: '{query[:60]}...' " 
              if len(query) > 60 else f"[ValidationLogger] Logged query: '{query}'")

    def save(self):
        """Write all logged entries to a JSON file."""
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump({"validation_log": self.entries}, f, indent=2, ensure_ascii=False)
        print(f"[ValidationLogger] Saved {len(self.entries)} entries to '{self.log_path}'.")

    def print_summary(self):
        """Print a formatted summary table of all logged queries."""
        if not self.entries:
            print("[ValidationLogger] No queries logged yet.")
            return
        headers = ["#", "Query (truncated)", "Top Score", "Avg Score", "Latency(s)", "Retrieved"]
        rows = []
        for i, e in enumerate(self.entries, 1):
            q = e["query"][:40] + "..." if len(e["query"]) > 40 else e["query"]
            rows.append([
                i,
                q,
                f"{e['top_score']:.4f}",
                f"{e['avg_score']:.4f}",
                f"{e['latency_seconds']}s",
                e["retrieved_count"],
            ])
        print("\n" + "=" * 70)
        print("  VALIDATION LOG SUMMARY")
        print("=" * 70)
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print(f"\n  Total queries logged: {len(self.entries)}")
        avg_latency = sum(e["latency_seconds"] for e in self.entries) / len(self.entries)
        print(f"  Average latency:      {avg_latency:.3f}s")
        avg_top = sum(e["top_score"] for e in self.entries) / len(self.entries)
        print(f"  Average top score:    {avg_top:.4f}")


def print_system_metrics(pipeline):
    """
    Print a comprehensive system metrics report.
    Covers chunking profiles, embedding dimensions, vector store setup, LLM config.
    """
    report = pipeline.get_system_report()

    print("\n" + "=" * 70)
    print("  SYSTEM METRICS REPORT")
    print("=" * 70)

    # ── Ingestion ──────────────────────────────────────────────────────────
    print("\n  [1] DOCUMENT INGESTION STATS")
    ing = report["ingestion_stats"]
    rows = [
        ["Total documents loaded", ing["total_documents"]],
        ["Total characters ingested", f"{ing['total_characters']:,}"],
        ["Unique sources", len(ing["unique_sources"])],
    ]
    print(tabulate(rows, tablefmt="simple"))
    for src in ing["unique_sources"]:
        print(f"       → {src}")

    # ── Chunking ───────────────────────────────────────────────────────────
    print("\n  [2] CHUNKING PROFILE")
    ch = report["chunking_config"]
    rows = [
        ["Strategy", ch["strategy"]],
        ["Chunk size (chars)", ch["chunk_size"]],
        ["Chunk overlap (chars)", ch["chunk_overlap"]],
    ]
    vs = report["vector_store_stats"]
    if isinstance(vs, dict):
        rows.append(["Total chunks created", vs["total_vectors"]])
    print(tabulate(rows, tablefmt="simple"))

    # ── Embeddings ─────────────────────────────────────────────────────────
    print("\n  [3] TEXT EMBEDDING CONFIGURATION")
    em = report["embedding_config"]
    rows = [
        ["Model name", em["model_name"]],
        ["Embedding dimension", em["embedding_dimension"]],
        ["Batch size", em["batch_size"]],
        ["Normalization", em["normalization"]],
    ]
    print(tabulate(rows, tablefmt="simple"))

    # ── Vector Store ───────────────────────────────────────────────────────
    print("\n  [4] VECTOR STORE (FAISS)")
    if isinstance(vs, dict):
        rows = [
            ["Total vectors stored", vs["total_vectors"]],
            ["Vector dimension", vs["dimension"]],
            ["Index type", vs["index_type"]],
            ["Description", vs["index_description"]],
        ]
        print(tabulate(rows, tablefmt="simple"))
    else:
        print(f"    Status: {vs}")

    # ── Retrieval ──────────────────────────────────────────────────────────
    print("\n  [5] RETRIEVAL CONFIGURATION")
    ret = report["retrieval_stats"]
    if isinstance(ret, dict):
        rows = [
            ["top_k", ret["top_k"]],
            ["Score threshold", ret["score_threshold"]],
            ["Hybrid search enabled", ret["hybrid_search_enabled"]],
            ["Hybrid alpha (vector weight)", ret["hybrid_alpha"]],
            ["Total queries processed", ret["queries_processed"]],
        ]
        print(tabulate(rows, tablefmt="simple"))

    # ── Generator / LLM ───────────────────────────────────────────────────
    print("\n  [6] LANGUAGE MODEL SETUP")
    gen = report["generator_config"]
    rows = [
        ["Model name", gen["model_name"]],
        ["Max new tokens", gen["max_new_tokens"]],
        ["Max context chars", gen["max_context_chars"]],
        ["Decoding strategy", gen["decoding_strategy"]],
        ["Device", gen.get("device", "N/A")],
    ]
    print(tabulate(rows, tablefmt="simple"))
    print("\n" + "=" * 70 + "\n")
