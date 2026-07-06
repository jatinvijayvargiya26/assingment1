"""
main.py — Week 7 Assignment Entry Point
=========================================
Document Question Answering System using Retrieval-Augmented Generation (RAG)

This script demonstrates the complete 8-step RAG pipeline:
  1. Document Ingestion
  2. Text Chunking
  3. Embedding Creation
  4. Vector Store Initialization
  5. Query Embedding
  6. Context Retrieval
  7. Answer Generation
  8. System Optimizations (Hybrid Search)

Run:
    python main.py                    # Demo mode with sample document
    python main.py --pdf my_file.pdf  # Use your own PDF
    python main.py --txt my_file.txt  # Use your own text file
    python main.py --hf squad         # Use HuggingFace SQuAD dataset
    python main.py --interactive      # Interactive Q&A mode

Author: Week 7 Assignment
"""

import os
import sys
import time
import argparse

# ── Rich output via colorama ──────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORED = True
except ImportError:
    COLORED = False
    class _Stub:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BLUE = RESET = ""
    Fore = _Stub()
    Style = _Stub()

# ── Pipeline imports ──────────────────────────────────────────────────────────
from rag_system import RAGPipeline
from validation_log import ValidationLogger, print_system_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Sample Queries for Demo/Validation
# ─────────────────────────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG) and how does it work?",
    "What are the advantages of RAG over purely generative models?",
    "Explain the difference between supervised and unsupervised learning.",
    "What is FAISS and why is it used in RAG systems?",
    "What is the T5 model and how does it treat NLP tasks?",
    "What are the stages of the RAG pipeline?",
    "What is the difference between BERT and GPT architectures?",
    "What are common applications of RAG systems?",
]


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def banner():
    print(f"\n{Fore.CYAN}{'=' * 65}")
    print(f"  [*] Week 7 Assignment - Document Question Answering System")
    print(f"       Retrieval-Augmented Generation (RAG) Pipeline")
    print(f"{'=' * 65}{Style.RESET_ALL}\n")


def build_pipeline(args) -> RAGPipeline:
    """Configure and return the RAG pipeline based on CLI args."""
    pipeline = RAGPipeline(
        chunk_size=512,
        chunk_overlap=128,
        chunk_strategy="recursive",
        embedding_model_name="all-MiniLM-L6-v2",
        top_k=5,
        score_threshold=0.0,
        use_hybrid_search=args.hybrid,
        hybrid_alpha=0.7,
        generator_model_name="google/flan-t5-base",
        max_new_tokens=256,
        verbose=True,
    )
    return pipeline


def ingest_sources(pipeline: RAGPipeline, args):
    """Ingest documents based on CLI arguments."""
    ingested = False

    if args.txt:
        if not os.path.exists(args.txt):
            print(f"{Fore.RED}[ERROR] File not found: {args.txt}{Style.RESET_ALL}")
            sys.exit(1)
        print(f"\n{Fore.YELLOW}[INFO] Ingesting TXT file: {args.txt}{Style.RESET_ALL}")
        pipeline.ingest_txt(args.txt)
        ingested = True

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"{Fore.RED}[ERROR] PDF not found: {args.pdf}{Style.RESET_ALL}")
            sys.exit(1)
        print(f"\n{Fore.YELLOW}[INFO] Ingesting PDF: {args.pdf}{Style.RESET_ALL}")
        pipeline.ingest_pdf(args.pdf)
        ingested = True

    if args.hf:
        print(f"\n{Fore.YELLOW}[INFO] Loading HuggingFace dataset: {args.hf}{Style.RESET_ALL}")
        pipeline.ingest_huggingface(
            dataset_name=args.hf,
            split="train",
            text_column="context",
            max_samples=args.hf_samples,
        )
        ingested = True

    if not ingested:
        # Default: use included sample document
        sample_path = os.path.join(os.path.dirname(__file__), "sample_docs", "sample.txt")
        if not os.path.exists(sample_path):
            print(f"{Fore.RED}[ERROR] Sample document not found at '{sample_path}'.{Style.RESET_ALL}")
            sys.exit(1)
        print(f"\n{Fore.YELLOW}[INFO] No source specified — using built-in sample document.{Style.RESET_ALL}")
        print(f"  → {sample_path}")
        print(f"  (Tip: use --txt, --pdf, or --hf flags to use your own documents)\n")
        pipeline.ingest_txt(sample_path)


def run_demo(pipeline: RAGPipeline, logger: ValidationLogger):
    """Run the pre-defined demo questions for validation."""
    print(f"\n{Fore.CYAN}{'-' * 65}")
    print(f"  Running {len(DEMO_QUESTIONS)} validation queries...")
    print(f"{'-' * 65}{Style.RESET_ALL}")

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{Fore.MAGENTA}[Query {i}/{len(DEMO_QUESTIONS)}]{Style.RESET_ALL}")
        t_start = time.time()
        result = pipeline.query(question, show_sources=True)
        latency = time.time() - t_start

        logger.log_query(
            query=question,
            answer=result["answer"],
            retrieved_chunks=result["retrieved_chunks"],
            scores=result["scores"],
            sources=result["sources"],
            latency_sec=latency,
        )
        # Small pause to avoid rapid model loading issues
        time.sleep(0.1)


def run_interactive(pipeline: RAGPipeline, logger: ValidationLogger):
    """Launch an interactive Q&A session."""
    print(f"\n{Fore.CYAN}{'-' * 65}")
    print(f"  Interactive Q&A Mode - type your question and press Enter")
    print(f"  Type 'quit' or 'exit' to stop. Type 'metrics' for system report.")
    print(f"{'-' * 65}{Style.RESET_ALL}\n")

    while True:
        try:
            question = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[INFO] Exiting interactive mode.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("[INFO] Goodbye!")
            break
        if question.lower() == "metrics":
            print_system_metrics(pipeline)
            continue

        t_start = time.time()
        try:
            result = pipeline.query(question, show_sources=True)
            latency = time.time() - t_start
            logger.log_query(
                query=question,
                answer=result["answer"],
                retrieved_chunks=result["retrieved_chunks"],
                scores=result["scores"],
                sources=result["sources"],
                latency_sec=latency,
            )
        except Exception as e:
            print(f"{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Week 7: Document Question Answering System (RAG)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--txt", type=str, default=None, help="Path to a .txt file to ingest.")
    parser.add_argument("--pdf", type=str, default=None, help="Path to a PDF file to ingest.")
    parser.add_argument("--hf", type=str, default=None,
                        help="HuggingFace dataset name (e.g., 'squad'). Uses 'context' column.")
    parser.add_argument("--hf-samples", type=int, default=200,
                        help="Max samples to load from HuggingFace dataset (default: 200).")
    parser.add_argument("--hybrid", action="store_true",
                        help="Enable hybrid search (BM25 + vector). Step 8 optimization.")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive Q&A mode after indexing.")
    parser.add_argument("--no-demo", action="store_true",
                        help="Skip the auto demo queries (useful with --interactive).")
    parser.add_argument("--save-index", type=str, default=None,
                        help="Directory to save the FAISS index for reuse.")
    return parser.parse_args()


def main():
    banner()
    args = parse_args()

    # ── 1. Build Pipeline ──────────────────────────────────────────────────
    pipeline = build_pipeline(args)

    # ── 2. Ingest Documents ────────────────────────────────────────────────
    ingest_sources(pipeline, args)

    # ── 3. Build FAISS Index ───────────────────────────────────────────────
    if args.save_index:
        pipeline.index_dir = args.save_index
    pipeline.build_index()

    # ── 4. System Metrics Report ───────────────────────────────────────────
    print_system_metrics(pipeline)

    # ── 5. Validation Demo Queries ─────────────────────────────────────────
    logger = ValidationLogger(log_path="validation_log.json")

    if not args.no_demo:
        run_demo(pipeline, logger)

    # ── 6. Interactive Mode ────────────────────────────────────────────────
    if args.interactive:
        run_interactive(pipeline, logger)

    # ── 7. Save Validation Log ─────────────────────────────────────────────
    logger.save()
    logger.print_summary()

    print(f"\n{Fore.GREEN}[DONE] Pipeline complete! Validation log saved to 'validation_log.json'.{Style.RESET_ALL}")
    print(f"  You can re-run with --interactive for live Q&A mode.")
    print(f"  Use --hybrid for hybrid BM25 + vector search (Step 8 optimization).\n")


if __name__ == "__main__":
    main()
