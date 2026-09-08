"""
Streamlit entry point for the AI Research Assistant.

Features
--------
✓ Chat with Paper
✓ Paper Analysis
✓ Compare Two Papers
✓ Paper Recommendations
✓ Debug Mode

Run:

streamlit run app.py
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from analysis.paper_analyzer import PaperAnalysisResult
from config.settings import get_settings
from core.pipeline import ResearchAssistantPipeline
from ingestion.pdf_loader import PDFLoadError
from llm.provider_router import AllProvidersFailedError
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="AI Research Scientist",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> ResearchAssistantPipeline:
    settings = get_settings()
    return ResearchAssistantPipeline(settings)


def init_session_state():

    st.session_state.setdefault("messages", [])

    st.session_state.setdefault("paper_ingested", False)
    st.session_state.setdefault("paper_title", None)

    st.session_state.setdefault("_last_uploaded_name", None)

    st.session_state.setdefault("paper_analysis", None)

    st.session_state.setdefault("paper_b_uploaded", False)
    st.session_state.setdefault("paper_b_title", None)
    st.session_state.setdefault("paper_b_name", None)
    st.session_state.setdefault("comparison_result", None)




def render_sidebar(pipeline):

    settings = get_settings()

    with st.sidebar:

        st.header("📄 AI Research Scientist")

        st.caption(
            "Chat • Analyze • Compare • Recommend"
        )

        st.subheader("Upload Paper A")

        uploaded_file = st.file_uploader(
            "Research Paper (PDF)",
            type=["pdf"],
            key="paper_a",
        )

        if uploaded_file is not None:

            already = (
                st.session_state.get("_last_uploaded_name")
                == uploaded_file.name
            )

            if not already:

                with st.spinner("Processing Paper A..."):

                    try:

                        metadata = pipeline.ingest_pdf(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                        )

                        st.session_state.paper_ingested = True
                        st.session_state.paper_title = metadata.title
                        st.session_state.messages = []
                        st.session_state.paper_analysis = None

                        st.session_state._last_uploaded_name = (
                            uploaded_file.name
                        )

                        st.success(
                            f"Indexed:\n\n{metadata.title}"
                        )

                    except PDFLoadError as exc:

                        st.error(str(exc))

                    except Exception as exc:

                        logger.exception(exc)

                        st.error(str(exc))

        if st.session_state.paper_ingested:

            st.success(
                f"Current Paper\n\n{st.session_state.paper_title}"
            )

        st.divider()

        st.subheader("Generation")

        provider = st.selectbox(
            "Preferred Provider",
            settings.app.llm.provider_priority,
        )

        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            settings.app.llm.temperature,
            step=0.05,
        )

        top_k = st.slider(
            "Top K",
            1,
            15,
            settings.app.retrieval.top_k,
        )

        debug = st.toggle(
            "Debug Mode",
            False,
        )

        st.divider()

        if st.button("Clear Chat"):

            st.session_state.messages = []

            st.rerun()

    return {
        "provider": provider,
        "temperature": temperature,
        "top_k": top_k,
        "debug": debug,
    }


def render_debug_panel(debug):

    with st.expander("Debug"):

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Provider",
            debug.get("provider_used") or "-"
        )

        c2.metric(
            "Latency",
            f"{debug.get('latency_seconds',0):.2f}s",
        )

        c3.metric(
            "Cache",
            "Yes" if debug.get("from_cache") else "No",
        )

        st.divider()

        chunks = debug.get("chunks") or []

        st.write(f"Retrieved {len(chunks)} chunks")

        for i, chunk in enumerate(chunks, start=1):

            st.markdown(
                f"### Chunk {i} | Score = {chunk.similarity_score}"
            )

            st.write(chunk.text[:500])

def render_chat(pipeline, sidebar):

    st.header("💬 Chat with your Research Paper")

    if not st.session_state.paper_ingested:
        st.info("Upload a research paper first.")
        return

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and sidebar["debug"]
                and message.get("debug")
            ):
                render_debug_panel(message["debug"])

    question = st.chat_input(
        "Ask anything about the uploaded paper..."
    )

    if question is None:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        holder = st.empty()
        holder.markdown("Thinking...")

        original_priority = (
            pipeline.settings.app.llm.provider_priority
        )

        preferred = sidebar["provider"]

        pipeline.settings.app.llm.provider_priority = [
            preferred,
            *[
                p
                for p in original_priority
                if p != preferred
            ],
        ]

        try:

            result = pipeline.ask(
                question=question,
                top_k=sidebar["top_k"],
                temperature=sidebar["temperature"],
            )

            holder.markdown(result.answer)

            debug = {
                "provider_used": result.provider_used,
                "latency_seconds": result.latency_seconds,
                "from_cache": result.from_cache,
                "chunks": result.retrieved_chunks,
            }

            if sidebar["debug"]:
                render_debug_panel(debug)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "debug": debug,
                }
            )

        except AllProvidersFailedError:

            holder.error(
                "All providers failed."
            )

        except Exception as exc:

            logger.exception(exc)

            holder.error(str(exc))

        finally:

            pipeline.settings.app.llm.provider_priority = (
                original_priority
            )


def render_paper_analysis(pipeline):

    st.header("📄 Paper Analysis")

    if not st.session_state.paper_ingested:

        st.info(
            "Upload a paper first."
        )

        return

    if st.button(
        "Analyze Paper",
        use_container_width=True,
    ):

        with st.spinner(
            "Analyzing paper..."
        ):

            try:

                analysis = pipeline.analyze_paper()

                st.session_state.paper_analysis = analysis

            except AllProvidersFailedError:

                st.error(
                    "All providers failed."
                )

            except Exception as exc:

                logger.exception(exc)

                st.error(str(exc))

    analysis: Optional[
        PaperAnalysisResult
    ] = st.session_state.paper_analysis

    if analysis is None:
        return

    st.caption(
        f"Provider: {analysis.provider_used} | "
        f"{analysis.latency_seconds:.2f}s | "
        f"{analysis.chunk_count} chunks"
    )

    st.markdown(analysis.analysis)
def render_recommendations(pipeline):

    st.header("🔍 Paper Recommendations")

    default_query = ""

    if (
        st.session_state.paper_ingested
        and st.session_state.paper_title
    ):
        default_query = st.session_state.paper_title

    query = st.text_input(
        "Search Query",
        value=default_query,
    )

    limit = st.slider(
        "Number of Papers",
        1,
        10,
        5,
    )

    if st.button(
        "Recommend Papers",
        use_container_width=True,
    ):

        if not query.strip():

            st.warning("Please enter a search query.")

            return

        with st.spinner(
            "Searching Semantic Scholar..."
        ):

            try:

                papers = pipeline.recommend_papers(
                    query=query,
                    limit=limit,
                )

                if not papers:

                    st.warning("No papers found.")

                    return

                for i, paper in enumerate(papers, start=1):

                    with st.expander(
                        f"{i}. {paper.title}",
                        expanded=False,
                    ):

                        st.markdown(
                            f"**Authors:** {paper.authors}"
                        )

                        st.markdown(
                            f"**Year:** {paper.year}"
                        )

                        st.markdown(
                            "**Abstract**"
                        )

                        st.write(paper.abstract)

                        if paper.url:

                            st.markdown(
                                f"[Open Paper]({paper.url})"
                            )

            except Exception as exc:

                logger.exception(exc)

                st.error(str(exc))


def render_comparison(pipeline):

    st.header("⚖️ Compare Two Papers")

    col1, col2 = st.columns(2)

    with col1:

        uploaded = st.file_uploader(
            "Upload Paper B",
            type=["pdf"],
            key="paper_b_upload",
        )

        if uploaded is not None:

            already = (
                st.session_state.get("paper_b_name")
                == uploaded.name
            )

            if not already:

                with st.spinner(
                    "Processing second paper..."
                ):

                    try:

                        metadata = pipeline.ingest_second_pdf(
                            uploaded.getvalue(),
                            uploaded.name,
                        )

                        st.session_state.paper_b_uploaded = True

                        st.session_state.paper_b_title = (
                            metadata.title
                        )

                        st.session_state.paper_b_name = (
                            uploaded.name
                        )

                        st.success(
                            metadata.title
                        )

                    except Exception as exc:

                        logger.exception(exc)

                        st.error(str(exc))

    with col2:

        if st.session_state.paper_b_uploaded:

            st.success(
                st.session_state.paper_b_title
            )

    if st.button(
        "Compare Papers",
        use_container_width=True,
    ):

        if not st.session_state.paper_ingested:

            st.warning(
                "Upload Paper A first."
            )

            return

        if not st.session_state.paper_b_uploaded:

            st.warning(
                "Upload Paper B first."
            )

            return

        with st.spinner(
            "Comparing papers..."
        ):

            try:

                result = pipeline.compare_papers()

                st.session_state.comparison_result = result

            except Exception as exc:

                logger.exception(exc)

                st.error(str(exc))

    result = st.session_state.get(
        "comparison_result"
    )

    if result is not None:

        st.caption(
            f"Provider: {result.provider_used} | "
            f"{result.latency_seconds:.2f}s"
        )

        st.markdown(result.comparison)


def main():

    init_session_state()

    pipeline = get_pipeline()

    sidebar = render_sidebar(pipeline)

    chat_tab, recommend_tab, compare_tab = st.tabs(
        [
            "💬 Chat",
            "🔍 Recommend",
            "⚖️ Compare",
        ]
    )

    with chat_tab:

        render_chat(
            pipeline,
            sidebar,
        )

        st.divider()

        render_paper_analysis(
            pipeline,
        )

    with recommend_tab:

        render_recommendations(
            pipeline,
        )

    with compare_tab:

        render_comparison(
            pipeline,
        )


if __name__ == "__main__":

    main()