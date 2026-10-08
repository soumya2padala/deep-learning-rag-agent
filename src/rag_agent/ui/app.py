"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
    - Left sidebar: Document ingestion and corpus browser
    - Centre: Document viewer
    - Right: Chat interface

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the cached VectorStoreManager."""
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the cached DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """Initialize Streamlit session state."""

    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """Render the document ingestion panel in the sidebar."""

    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md", "markdown"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "📥 Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            file_paths = []

            for uploaded_file in uploaded_files:
                destination = (
                    temp_path / uploaded_file.name
                )

                destination.write_bytes(
                    uploaded_file.getbuffer()
                )

                file_paths.append(destination)

            with st.spinner(
                "Processing documents..."
            ):
                chunks = chunker.chunk_files(
                    file_paths
                )

                result = store.ingest(
                    chunks
                )

        st.session_state[
            "last_ingestion_result"
        ] = result

        if result.ingested > 0:
            st.sidebar.success(
                f"{result.ingested} chunks added, "
                f"{result.skipped} duplicates skipped."
            )
        elif result.skipped > 0:
            st.sidebar.warning(
                f"0 new chunks added, "
                f"{result.skipped} duplicates skipped."
            )

        if result.errors:
            st.sidebar.error(
                f"{len(result.errors)} chunks "
                "could not be processed."
            )

        # Refresh document list.
        st.session_state[
            "ingested_documents"
        ] = store.list_documents()

        st.rerun()

    # Refresh documents every rerun.
    documents = store.list_documents()

    st.session_state[
        "ingested_documents"
    ] = documents

    if documents:
        st.sidebar.subheader(
            "📚 Ingested Documents"
        )

        for index, document in enumerate(
            documents
        ):
            source = document.get(
                "source",
                "Unknown",
            )

            topic = document.get(
                "topic",
                "Unknown",
            )

            chunk_count = document.get(
                "chunk_count",
                0,
            )

            col1, col2 = st.sidebar.columns(
                [4, 1]
            )

            with col1:
                st.caption(
                    f"**{source}**\n\n"
                    f"{topic} • "
                    f"{chunk_count} chunks"
                )

            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_{index}_{source}",
                    help=f"Delete {source}",
                ):
                    deleted = (
                        store.delete_document(
                            source
                        )
                    )

                    if (
                        st.session_state[
                            "selected_document"
                        ]
                        == source
                    ):
                        st.session_state[
                            "selected_document"
                        ] = None

                    st.sidebar.success(
                        f"Deleted {deleted} chunks."
                    )

                    st.rerun()

    else:
        st.sidebar.info(
            "Upload .pdf or .md files "
            "to populate the corpus."
        )


# ---------------------------------------------------------------------------
# Corpus Statistics
# ---------------------------------------------------------------------------


def render_corpus_stats(
    store: VectorStoreManager,
) -> None:
    """Render a compact corpus health summary."""

    stats = store.get_collection_stats()

    st.sidebar.subheader(
        "📊 Corpus Statistics"
    )

    st.sidebar.metric(
        "Total Chunks",
        stats["total_chunks"],
    )

    topics = stats.get(
        "topics",
        [],
    )

    if topics:
        st.sidebar.write(
            "Topics:",
            ", ".join(topics),
        )
    else:
        st.sidebar.write(
            "Topics: None yet"
        )

    if stats.get(
        "bonus_topics_present",
        False,
    ):
        st.sidebar.success(
            "✅ Bonus topics present"
        )
    else:
        st.sidebar.warning(
            "⚠️ No bonus topics yet"
        )


# ---------------------------------------------------------------------------
# Document Viewer Panel
# ---------------------------------------------------------------------------


def render_document_viewer(
    store: VectorStoreManager,
) -> None:
    """Render the document viewer."""

    st.subheader(
        "📄 Document Viewer"
    )

    documents = store.list_documents()

    if not documents:
        st.info(
            "Ingest documents using the "
            "sidebar to view content here."
        )
        return

    document_sources = [
        document["source"]
        for document in documents
    ]

    current_selection = (
        st.session_state.get(
            "selected_document"
        )
    )

    if (
        current_selection
        not in document_sources
    ):
        current_selection = (
            document_sources[0]
        )

    selected_source = st.selectbox(
        "Select document",
        options=document_sources,
        index=document_sources.index(
            current_selection
        ),
    )

    st.session_state[
        "selected_document"
    ] = selected_source

    chunks = store.get_document_chunks(
        selected_source
    )

    st.caption(
        f"**{len(chunks)} chunks** "
        f"from `{selected_source}`"
    )

    if not chunks:
        st.warning(
            "No chunks found for this document."
        )
        return

    with st.container(height=550):
        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            metadata = chunk.metadata

            topic = getattr(
                metadata,
                "topic",
                "Unknown",
            )

            difficulty = getattr(
                metadata,
                "difficulty",
                "Unknown",
            )

            is_bonus = getattr(
                metadata,
                "is_bonus",
                False,
            )

            badge = (
                "⭐ Bonus"
                if is_bonus
                else "📘 Standard"
            )

            st.markdown(
                f"**Chunk {index}** · "
                f"`{topic}` · "
                f"`{difficulty}` · "
                f"{badge}"
            )

            st.markdown(
                chunk.chunk_text
            )

            if index < len(chunks):
                st.divider()


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------


def _get_response_value(
    response,
    name: str,
    default=None,
):
    """Safely get a value from a response object or dictionary."""

    if isinstance(response, dict):
        return response.get(
            name,
            default,
        )

    return getattr(
        response,
        name,
        default,
    )


def _format_sources(
    sources,
) -> list[str]:
    """Convert source information into displayable strings."""

    if not sources:
        return []

    formatted = []

    for source in sources:
        if isinstance(
            source,
            str,
        ):
            formatted.append(source)

        elif isinstance(
            source,
            dict,
        ):
            formatted.append(
                str(source)
            )

        else:
            try:
                if hasattr(
                    source,
                    "to_citation",
                ):
                    formatted.append(
                        source.to_citation()
                    )
                else:
                    formatted.append(
                        str(source)
                    )
            except Exception:
                formatted.append(
                    str(source)
                )

    return formatted


def render_chat_interface(
    graph,
) -> None:
    """Render the RAG chat interface."""

    st.subheader(
        "💬 Interview Prep Chat"
    )

    # ---------------------------------------------------------------
    # Filters
    # ---------------------------------------------------------------

    topics = ["All"]

    documents = st.session_state.get(
        "ingested_documents",
        [],
    )

    for document in documents:
        topic = document.get(
            "topic"
        )

        if (
            topic
            and topic not in topics
        ):
            topics.append(topic)

    difficulties = [
        "All",
        "beginner",
        "intermediate",
        "advanced",
    ]

    col_topic, col_diff = st.columns(
        2
    )

    with col_topic:
        selected_topic = st.selectbox(
            "Topic",
            options=topics,
        )

    with col_diff:
        selected_difficulty = (
            st.selectbox(
                "Difficulty",
                options=difficulties,
            )
        )

    st.session_state[
        "topic_filter"
    ] = (
        None
        if selected_topic == "All"
        else selected_topic
    )

    st.session_state[
        "difficulty_filter"
    ] = (
        None
        if selected_difficulty == "All"
        else selected_difficulty
    )

    # ---------------------------------------------------------------
    # Chat history
    # ---------------------------------------------------------------

    chat_container = st.container(
        height=400
    )

    with chat_container:
        for message in (
            st.session_state.chat_history
        ):
            with st.chat_message(
                message["role"]
            ):
                st.markdown(
                    message["content"]
                )

                sources = message.get(
                    "sources",
                    [],
                )

                if sources:
                    with st.expander(
                        "📎 Sources"
                    ):
                        for source in sources:
                            st.caption(
                                source
                            )

                if message.get(
                    "no_context_found",
                    False,
                ):
                    st.warning(
                        "⚠️ No relevant content "
                        "found in corpus."
                    )

    # ---------------------------------------------------------------
    # Chat input
    # ---------------------------------------------------------------

    query = st.chat_input(
        "Ask about a deep learning topic..."
    )

    if not query:
        return

    # Add user message.
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    filters = {
        "topic": st.session_state.get(
            "topic_filter"
        ),
        "difficulty": st.session_state.get(
            "difficulty_filter"
        ),
    }

    graph_input = {
        "messages": [
            HumanMessage(
                content=query
            )
        ],
    }

    # Include filters if the graph state supports them.
    graph_input["filters"] = filters

    config = {
        "configurable": {
            "thread_id": (
                st.session_state[
                    "thread_id"
                ]
            )
        }
    }

    try:
        with st.spinner(
            "Searching the corpus and generating an answer..."
        ):
            result = graph.invoke(
                graph_input,
                config=config,
            )

        response = result.get(
            "final_response",
            result,
        )

        answer = _get_response_value(
            response,
            "answer",
            None,
        )

        if answer is None:
            answer = _get_response_value(
                response,
                "response",
                None,
            )

        if answer is None:
            answer = _get_response_value(
                response,
                "content",
                None,
            )

        if answer is None:
            answer = str(response)

        sources = _get_response_value(
            response,
            "sources",
            [],
        )

        no_context_found = (
            _get_response_value(
                response,
                "no_context_found",
                False,
            )
        )

        formatted_sources = (
            _format_sources(sources)
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": str(answer),
                "sources": formatted_sources,
                "no_context_found": (
                    no_context_found
                ),
            }
        )

        st.rerun()

    except Exception as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": (
                    "I encountered an error while "
                    f"processing your question: {exc}"
                ),
                "sources": [],
                "no_context_found": False,
            }
        )

        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""

    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(
        f"🧠 {settings.app_title}"
    )

    st.caption(
        "RAG-powered interview preparation — "
        "built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Shared backend resources.
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar.
    render_ingestion_panel(
        store,
        chunker,
    )

    render_corpus_stats(
        store
    )

    # Main content.
    viewer_col, chat_col = st.columns(
        [1, 1],
        gap="large",
    )

    with viewer_col:
        render_document_viewer(
            store
        )

    with chat_col:
        render_chat_interface(
            graph
        )


if __name__ == "__main__":
    main()