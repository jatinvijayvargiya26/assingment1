"""
Answer Generator Module
=======================
Step 7 of the RAG Pipeline.

Constructs a grounded prompt from retrieved context chunks + user query,
then uses a local Language Model (LM) to generate an answer.

Model Options (all free, run locally):
- google/flan-t5-base    (250MB, fast, good for Q&A) ← DEFAULT
- google/flan-t5-large   (780MB, better quality)
- distilgpt2             (very small, less accurate)

Approach:
  Context from Step 6 → Unified Prompt → LLM → Grounded Answer

Author: Week 7 Assignment
"""

from typing import List, Tuple, Dict, Optional
from .chunking import Chunk


class AnswerGenerator:
    """
    LLM-based answer generator using retrieved context chunks.

    Parameters
    ----------
    model_name : str
        HuggingFace model name. Default: 'google/flan-t5-base'
    max_new_tokens : int
        Maximum number of tokens in the generated answer.
    max_context_chars : int
        Maximum total characters of context to include in the prompt.
        Prevents prompt from exceeding model's context window.
    device : str
        'cpu' or 'cuda' (auto-detected if None)
    """

    DEFAULT_MODEL = "google/flan-t5-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 256,
        max_context_chars: int = 2000,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self._tokenizer = None
        self._model = None
        self._device = device
        print(f"[Generator] Model set to '{model_name}'. Will load on first use.")

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the HuggingFace model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
                import torch
            except ImportError:
                raise ImportError(
                    "transformers and torch not installed.\n"
                    "Run: pip install transformers torch"
                )

            import torch
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[Generator] Loading '{self.model_name}' on {self._device.upper()}...")
            print(f"[Generator]   (First run will download ~250MB for flan-t5-base)")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # flan-t5 variants are seq2seq; GPT variants are causal LM
            if "t5" in self.model_name.lower():
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self._is_seq2seq = True
            else:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self._is_seq2seq = False

            self._model = self._model.to(self._device)
            self._model.eval()
            print(f"[Generator] Model loaded. Device: {self._device.upper()}")

    # ── Prompt Construction (Step 7) ──────────────────────────────────────────

    def build_prompt(self, query: str, retrieved_chunks: List[Tuple[float, Chunk]]) -> str:
        """
        Step 7: Construct a unified prompt from retrieved context + query.

        Concatenates top retrieved chunks (truncated to max_context_chars)
        into a structured prompt for the LLM.

        Returns
        -------
        str — the full prompt string sent to the LLM.
        """
        # Collect context text, respecting character limit
        context_parts = []
        total_chars = 0
        for score, chunk in retrieved_chunks:
            chunk_text = chunk.text.strip()
            if total_chars + len(chunk_text) > self.max_context_chars:
                # Include partial chunk to fill remaining space
                remaining = self.max_context_chars - total_chars
                if remaining > 50:
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(chunk_text)
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        context = "\n\n---\n\n".join(context_parts)

        # Structured RAG prompt — instructs the model to answer ONLY from context
        prompt = (
            "You are a helpful assistant that answers questions based only on the provided context.\n"
            "If the answer is not found in the context, say 'I could not find this information in the document.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return prompt

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, query: str, retrieved_chunks: List[Tuple[float, Chunk]]) -> Dict:
        """
        Generate a grounded answer for the given query using retrieved chunks.

        Parameters
        ----------
        query : str
            The user's question.
        retrieved_chunks : List[(score, Chunk)]
            Top-k retrieved chunks from the Retriever.

        Returns
        -------
        dict with keys:
            'answer'  — the generated text
            'prompt'  — the full prompt used
            'sources' — list of source strings for the retrieved chunks
            'scores'  — list of similarity scores
        """
        self._load_model()

        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the documents.",
                "prompt": "",
                "sources": [],
                "scores": [],
            }

        # Build prompt
        prompt = self.build_prompt(query, retrieved_chunks)
        print(f"[Generator] Prompt built ({len(prompt)} chars). Generating answer...")

        # Tokenize
        import torch
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._device)

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,       # Greedy decoding for factual answers
                temperature=1.0,
                repetition_penalty=1.2,
            )

        # Decode
        if self._is_seq2seq:
            answer = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            # For causal models, skip the prompt tokens in the output
            answer = self._tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

        answer = answer.strip()
        print(f"[Generator] Answer generated ({len(answer)} chars).")

        # Build source attribution info
        sources = []
        scores = []
        for score, chunk in retrieved_chunks:
            src = f"{chunk.source}"
            if chunk.page:
                src += f", page {chunk.page}"
            src += f" (chunk #{chunk.chunk_id})"
            sources.append(src)
            scores.append(round(score, 4))

        return {
            "answer": answer,
            "prompt": prompt,
            "sources": sources,
            "scores": scores,
        }

    def get_config(self) -> Dict:
        """Return generator configuration for metrics report."""
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "max_context_chars": self.max_context_chars,
            "decoding_strategy": "Greedy (do_sample=False)",
            "device": self._device or "not loaded yet",
        }
