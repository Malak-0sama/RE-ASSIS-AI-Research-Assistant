"""
Prompt template for full-paper analysis.

Centralized here (never hardcoded inside the analyzer) so the analysis
structure and instructions can be tuned without touching orchestration code.
"""
from __future__ import annotations

PAPER_ANALYSIS_SYSTEM_INSTRUCTIONS = """You are a senior AI research scientist producing a complete, structured \
analysis of an uploaded research paper for both newcomers and technical experts.

Rules you MUST follow:
1. Base your analysis ONLY on the information contained in the CONTEXT below.
2. Do not use outside knowledge about the paper, its authors, or related work.
3. If a section cannot be determined from the CONTEXT, write "Not clearly stated in the paper" \
for that section instead of guessing or fabricating details.
4. Be specific and factual. Avoid vague filler language.
5. Use clear Markdown formatting: a "## " heading for each of the required sections below, in the \
exact order given, followed by concise prose and bullet points where appropriate."""

ANALYSIS_SECTIONS = [
    "Beginner Summary",
    "Technical Summary",
    "Problem Statement",
    "Main Contributions",
    "Methodology",
    "Models Used",
    "Dataset Used",
    "Experimental Results",
    "Advantages",
    "Limitations",
    "Novelty",
    "Future Work",
]


def build_paper_analysis_prompt(context: str) -> str:
    """Assemble the final prompt sent to the LLM for one full-paper analysis."""
    section_list = "\n".join(f"{i}. {name}" for i, name in enumerate(ANALYSIS_SECTIONS, start=1))
    return (
        f"{PAPER_ANALYSIS_SYSTEM_INSTRUCTIONS}\n\n"
        f"REQUIRED SECTIONS (in this order):\n{section_list}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANALYSIS:"
    )
