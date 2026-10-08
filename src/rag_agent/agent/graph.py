"""
graph.py
========
LangGraph agent graph definition and compilation.

Assembles the nodes from nodes.py into a directed state graph
and compiles it with a memory checkpointer for conversation persistence.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_agent.agent.nodes import (
    generation_node,
    query_rewrite_node,
    retrieval_node,
    should_retry_retrieval,
)
from rag_agent.agent.state import AgentState


class AgentGraphBuilder:
    """
    Constructs and compiles the LangGraph agent state graph.

    The graph implements a three-node RAG pipeline:

        [START]
           │
           ▼
    query_rewrite_node
           │
           ▼
    retrieval_node
           │
           ▼
     ┌─────┴──────┐
     │            │
  "generate"    "end"
     │            │
     ▼            ▼
generation_node  [END]
     │
     ▼
   [END]
    """

    def __init__(self) -> None:
        self._checkpointer = MemorySaver()

    def build(self):
        """
        Assemble nodes and edges, then compile the graph.
        """

        # Create the state graph.
        graph = StateGraph(
            AgentState
        )

        # Add the RAG pipeline nodes.
        graph.add_node(
            "query_rewrite",
            query_rewrite_node,
        )

        graph.add_node(
            "retrieval",
            retrieval_node,
        )

        graph.add_node(
            "generation",
            generation_node,
        )

        # Start -> query rewriting.
        graph.add_edge(
            START,
            "query_rewrite",
        )

        # Query rewriting -> retrieval.
        graph.add_edge(
            "query_rewrite",
            "retrieval",
        )

        # Retrieval -> generation or END.
        #
        # If relevant context was found:
        #     "generate" -> generation node
        #
        # If no relevant context was found:
        #     "end" -> END
        graph.add_conditional_edges(
            "retrieval",
            should_retry_retrieval,
            {
                "generate": "generation",
                "end": END,
            },
        )

        # Generation -> END.
        graph.add_edge(
            "generation",
            END,
        )

        # Compile with MemorySaver so conversation
        # state can persist by thread_id.
        return graph.compile(
            checkpointer=self._checkpointer
        )


@lru_cache(maxsize=1)
def get_compiled_graph():
    """
    Return the singleton compiled graph.

    Uses lru_cache so the graph is built only once per process.
    """

    return AgentGraphBuilder().build()