"""
state.py
========
Data models and agent state definitions for the LangGraph agent.

All inter-component data structures are defined here to ensure
consistent interfaces across the entire codebase.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


# ---------------------------------------------------------------------------
# Corpus / Ingestion Models
# ---------------------------------------------------------------------------


@dataclass
class ChunkMetadata:
    """
    Metadata attached to every chunk stored in ChromaDB.
    """

    topic: str
    difficulty: str
    type: str
    source: str
    related_topics: list[str] = field(
        default_factory=list
    )
    is_bonus: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict for ChromaDB metadata storage."""

        return {
            "topic": self.topic,
            "difficulty": self.difficulty,
            "type": self.type,
            "source": self.source,
            "related_topics": ",".join(
                self.related_topics
            ),
            "is_bonus": str(
                self.is_bonus
            ).lower(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> ChunkMetadata:
        """Deserialise from a ChromaDB metadata dict."""

        related = data.get(
            "related_topics",
            "",
        )

        is_bonus = data.get(
            "is_bonus",
            "false",
        )

        if isinstance(
            is_bonus,
            bool,
        ):
            bonus_value = is_bonus
        else:
            bonus_value = (
                str(is_bonus).lower()
                == "true"
            )

        return cls(
            topic=data["topic"],
            difficulty=data["difficulty"],
            type=data["type"],
            source=data["source"],
            related_topics=(
                related.split(",")
                if related
                else []
            ),
            is_bonus=bonus_value,
        )


@dataclass
class DocumentChunk:
    """
    A single unit of content ready for embedding and storage.
    """

    chunk_id: str
    chunk_text: str
    metadata: ChunkMetadata


@dataclass
class IngestionResult:
    """
    Summary of a single ingestion operation.
    """

    ingested: int = 0
    skipped: int = 0
    errors: list[str] = field(
        default_factory=list
    )
    document_ids: list[str] = field(
        default_factory=list
    )

    @property
    def total_processed(self) -> int:
        """Total chunks processed."""

        return (
            self.ingested
            + self.skipped
            + len(self.errors)
        )

    @property
    def success(self) -> bool:
        """True if at least one chunk was ingested without error."""

        return (
            self.ingested > 0
            and len(self.errors) == 0
        )


# ---------------------------------------------------------------------------
# Retrieval Models
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """
    A single chunk returned from a vector store query.
    """

    chunk_id: str
    chunk_text: str
    metadata: ChunkMetadata
    score: float

    def to_citation(self) -> str:
        """Format this chunk as a source citation string."""

        return (
            f"[{self.metadata.topic} | "
            f"{self.metadata.difficulty} | "
            f"{self.metadata.source}]"
        )


# ---------------------------------------------------------------------------
# Agent Response Models
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """
    The structured response returned by the LangGraph agent.
    """

    answer: str
    sources: list[str] = field(
        default_factory=list
    )
    confidence: float = 0.0
    no_context_found: bool = False
    rewritten_query: str = ""


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class AgentState(MessagesState):
    """
    The state object passed between all nodes in the LangGraph agent.

    Inherits from MessagesState which provides the messages field.
    """

    original_query: str = ""

    rewritten_query: str = ""

    retrieved_chunks: list[
        RetrievedChunk
    ] = field(
        default_factory=list
    )

    no_context_found: bool = False

    final_response: AgentResponse | None = None

    topic_filter: str | None = None

    difficulty_filter: str | None = None