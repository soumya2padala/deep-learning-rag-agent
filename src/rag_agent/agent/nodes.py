"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import (
    AgentResponse,
    AgentState,
)
from rag_agent.config import (
    LLMFactory,
    get_settings,
)
from rag_agent.vectorstore.store import (
    VectorStoreManager,
)


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(
    state: AgentState,
) -> dict:
    """
    Rewrite the user's query for better vector retrieval.
    """

    messages = state.get(
        "messages",
        [],
    )

    original_query = ""

    # Find the latest human message.
    for message in reversed(messages):
        if isinstance(
            message,
            HumanMessage,
        ):
            original_query = str(
                message.content
            )
            break

    # Fallbacks.
    if not original_query:
        original_query = state.get(
            "original_query",
            "",
        )

    if not original_query:
        return {
            "original_query": "",
            "rewritten_query": "",
        }

    try:
        settings = get_settings()
        llm = LLMFactory(
            settings
        ).create()

        rewrite_prompt = (
            "Rewrite the following user question "
            "for semantic vector search over a "
            "deep learning study corpus.\n\n"
            "Keep the meaning unchanged. "
            "Use concise technical keywords. "
            "Return only the rewritten search query.\n\n"
            f"User question: {original_query}"
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a query rewriting "
                        "assistant for a deep learning "
                        "RAG system."
                    )
                ),
                HumanMessage(
                    content=rewrite_prompt
                ),
            ]
        )

        rewritten = str(
            response.content
        ).strip()

        if not rewritten:
            rewritten = original_query

    except Exception:
        # If the LLM rewrite fails, continue
        # using the original question.
        rewritten = original_query

    return {
        "original_query": original_query,
        "rewritten_query": rewritten,
    }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(
    state: AgentState,
) -> dict:
    """
    Retrieve relevant chunks from ChromaDB.
    """

    query_text = (
        state.get(
            "rewritten_query",
            "",
        )
        or state.get(
            "original_query",
            "",
        )
    )

    if not query_text:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    try:
        manager = VectorStoreManager()

        chunks = manager.query(
            query_text=query_text,
            topic_filter=state.get(
                "topic_filter"
            ),
            difficulty_filter=state.get(
                "difficulty_filter"
            ),
        )

    except Exception:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    if not chunks:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    return {
        "retrieved_chunks": chunks,
        "no_context_found": False,
    }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(
    state: AgentState,
) -> dict:
    """
    Generate the final response using retrieved chunks.
    """

    settings = get_settings()
    llm = LLMFactory(
        settings
    ).create()

    rewritten_query = state.get(
        "rewritten_query",
        "",
    )

    # ---- Hallucination Guard -----------------------------------------------

    no_context_found = state.get(
        "no_context_found",
        False,
    )

    if no_context_found:
        no_context_message = (
            "I was unable to find relevant information "
            "in the corpus for your query. This may mean "
            "the topic is not yet covered in the study "
            "material, or your query may need to be "
            "rephrased. Please try a more specific "
            "deep learning topic such as "
            "'LSTM forget gate' or "
            "'CNN pooling layers'."
        )

        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=rewritten_query,
        )

        return {
            "final_response": response,
            "messages": [
                AIMessage(
                    content=no_context_message
                )
            ],
        }

    # ---- Build Context ------------------------------------------------------

    retrieved_chunks = state.get(
        "retrieved_chunks",
        [],
    )

    context_parts = []
    sources = []
    scores = []

    for chunk in retrieved_chunks:
        citation = chunk.to_citation()

        context_parts.append(
            f"[SOURCE: {citation}]\n"
            f"{chunk.chunk_text}\n"
        )

        sources.append(citation)

        scores.append(
            float(chunk.score)
        )

    context = "\n".join(
        context_parts
    )

    confidence = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    # ---- Conversation History ----------------------------------------------

    conversation_messages = list(
        state.get(
            "messages",
            [],
        )
    )

    try:
        trimmed_messages = trim_messages(
            conversation_messages,
            max_tokens=settings.max_context_tokens,
            strategy="last",
            token_counter=llm,
            include_system=False,
            allow_partial=False,
        )

    except Exception:
        # Fallback if token counting is unsupported.
        trimmed_messages = (
            conversation_messages[-6:]
        )

    # ---- Build Final Prompt -------------------------------------------------

    context_message = (
        "Use the following retrieved study "
        "material to answer the user's question. "
        "Base your answer on this context.\n\n"
        "Retrieved context:\n"
        f"{context}"
    )

    prompt_messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=context_message
        ),
    ]

    prompt_messages.extend(
        trimmed_messages
    )

    original_query = state.get(
        "original_query",
        "",
    )

    if not original_query:
        original_query = rewritten_query

    prompt_messages.append(
        HumanMessage(
            content=original_query
        )
    )

    # ---- Generate Answer ---------------------------------------------------

    try:
        llm_response = llm.invoke(
            prompt_messages
        )

        answer = str(
            llm_response.content
        ).strip()

    except Exception as exc:
        answer = (
            "I encountered an error while "
            "generating the answer. "
            f"Please try again. ({exc})"
        )

    # ---- Structured Response ------------------------------------------------

    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=rewritten_query,
    )

    return {
        "final_response": response,
        "messages": [
            AIMessage(
                content=answer
            )
        ],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(
    state: AgentState,
) -> str:
    """
    Decide whether the graph should proceed to generation.

    The generation node contains the hallucination guard,
    so we always route there after retrieval.
    """

    if state.get(
        "no_context_found",
        False,
    ):
        return "generate"

    return "generate"