"""
Prompt templates for the RAG question-answering pipeline.

Prompts are centralized here (never hardcoded inline in pipeline logic) so
they can be tuned without touching orchestration code.
"""
from __future__ import annotations

FALLBACK_ANSWER = "I couldn't find this information in the uploaded paper."

QA_SYSTEM_INSTRUCTIONS = f"""You are a precise research assistant answering questions about a single \
uploaded research paper.

Rules you MUST follow:
1. Answer ONLY using the information contained in the CONTEXT below.
2. Do not use any outside knowledge, even if you are confident it is correct.
3. If the CONTEXT does not contain enough information to answer the question, \
respond with exactly: "{FALLBACK_ANSWER}"
4. Be concise and factual. Where useful, reference which source segment \
supports your answer (e.g. "According to Source 2...").
5. Do not fabricate citations, numbers, or claims that are not present in the CONTEXT."""


def build_qa_prompt(context: str, question: str) -> str:
    """Assemble the final prompt sent to the LLM for one QA turn."""
    return (
        f"{QA_SYSTEM_INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:"
    )
