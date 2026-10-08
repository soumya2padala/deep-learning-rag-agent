from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB vector store.

    Responsibilities:
    - Initialize ChromaDB
    - Generate deterministic chunk IDs
    - Detect duplicate chunks
    - Ingest document chunks
    - Query similar chunks
    - List documents
    - Retrieve document chunks
    - Provide collection statistics
    - Delete documents
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the vector store manager."""

        self._settings = settings or get_settings()

        self._embeddings = EmbeddingFactory(
            self._settings
        ).create()

        self._client = None
        self._collection = None

        self._initialise()

    # ------------------------------------------------------------------
    # ChromaDB initialization
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        """Initialize the persistent ChromaDB client and collection."""

        db_path = Path(
            self._settings.chroma_db_path
        )

        db_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            self._client = chromadb.PersistentClient(
                path=str(db_path)
            )

            self._collection = (
                self._client.get_or_create_collection(
                    name=self._settings.chroma_collection_name,
                    metadata={
                        "hnsw:space": "cosine"
                    },
                )
            )

            logger.info(
                "ChromaDB initialized: collection='{}', items={}",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )

        except Exception as exc:
            logger.exception(
                "Failed to initialize ChromaDB"
            )

            raise RuntimeError(
                f"Unable to initialize ChromaDB: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Chunk ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(
        source: str,
        chunk_text: str,
    ) -> str:
        """
        Generate a deterministic ID for a chunk.

        The same source and chunk text will always
        generate the same ID.
        """

        content = f"{source}::{chunk_text}"

        return hashlib.sha256(
            content.encode()
        ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def check_duplicate(
        self,
        chunk_id: str,
    ) -> bool:
        """Check whether a chunk already exists in ChromaDB."""

        result = self._collection.get(
            ids=[chunk_id]
        )

        return chunk_id in result.get(
            "ids",
            [],
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        chunks: list[DocumentChunk],
    ) -> IngestionResult:
        """
        Ingest document chunks into ChromaDB.

        Duplicate chunks are skipped.
        New chunks are embedded and stored.
        """

        result = IngestionResult()

        if not chunks:
            return result

        # Process chunks in batches of 100.
        for start in range(
            0,
            len(chunks),
            100,
        ):
            batch = chunks[
                start : start + 100
            ]

            for chunk in batch:
                try:
                    # Check for duplicates.
                    if self.check_duplicate(
                        chunk.chunk_id
                    ):
                        result.skipped += 1
                        continue

                    # Generate embedding.
                    embedding = (
                        self._embeddings.embed_documents(
                            [chunk.chunk_text]
                        )[0]
                    )

                    # Store the chunk.
                    self._collection.upsert(
                        ids=[
                            chunk.chunk_id
                        ],
                        embeddings=[
                            embedding
                        ],
                        documents=[
                            chunk.chunk_text
                        ],
                        metadatas=[
                            chunk.metadata.to_dict()
                        ],
                    )

                    result.ingested += 1

                except Exception as exc:
                    logger.exception(
                        "Failed to ingest chunk {}",
                        chunk.chunk_id,
                    )

                    result.errors.append(
                        f"{chunk.chunk_id}: {exc}"
                    )

        logger.info(
            "Ingestion complete: ingested={}, skipped={}, errors={}",
            result.ingested,
            result.skipped,
            len(result.errors),
        )

        return result

    # ------------------------------------------------------------------
    # Query / Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Query ChromaDB for the most similar chunks.
        """

        k = (
            k
            or self._settings.retrieval_k
        )

        where_filter = None

        filters = []

        if topic_filter:
            filters.append(
                {
                    "topic": topic_filter
                }
            )

        if difficulty_filter:
            filters.append(
                {
                    "difficulty": difficulty_filter
                }
            )

        if len(filters) == 1:
            where_filter = filters[0]

        elif len(filters) > 1:
            where_filter = {
                "$and": filters
            }

        # Generate embedding for the query.
        query_embedding = (
            self._embeddings.embed_query(
                query_text
            )
        )

        query_kwargs = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where_filter:
            query_kwargs["where"] = (
                where_filter
            )

        results = self._collection.query(
            **query_kwargs
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        ids = results.get(
            "ids",
            [[]],
        )[0]

        retrieved = []

        for (
            chunk_id,
            document,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            # ChromaDB is configured for cosine distance.
            # Similarity = 1 - distance.
            score = (
                1.0 - float(distance)
            )

            # Ignore results below the configured
            # similarity threshold.
            if (
                score
                < self._settings.similarity_threshold
            ):
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=ChunkMetadata.from_dict(
                        metadata
                    ),
                    score=score,
                )
            )

        # Highest similarity first.
        retrieved.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return retrieved

    # ------------------------------------------------------------------
    # List documents
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """List all documents stored in ChromaDB."""

        result = self._collection.get(
            include=["metadatas"]
        )

        metadatas = result.get(
            "metadatas",
            [],
        )

        documents = {}

        for metadata in metadatas:
            source = metadata.get(
                "source",
                "unknown",
            )

            topic = metadata.get(
                "topic",
                "unknown",
            )

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": topic,
                    "chunk_count": 0,
                }

            documents[source][
                "chunk_count"
            ] += 1

        return sorted(
            documents.values(),
            key=lambda item: item["source"],
        )

    # ------------------------------------------------------------------
    # Get document chunks
    # ------------------------------------------------------------------

    def get_document_chunks(
        self,
        source: str,
    ) -> list[DocumentChunk]:
        """Retrieve all chunks belonging to a document."""

        result = self._collection.get(
            where={
                "source": source
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = result.get(
            "documents",
            [],
        )

        metadatas = result.get(
            "metadatas",
            [],
        )

        ids = result.get(
            "ids",
            [],
        )

        chunks = []

        for (
            chunk_id,
            document,
            metadata,
        ) in zip(
            ids,
            documents,
            metadatas,
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=ChunkMetadata.from_dict(
                        metadata
                    ),
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Collection statistics
    # ------------------------------------------------------------------

    def get_collection_stats(
        self,
    ) -> dict:
        """Return statistics about the ChromaDB collection."""

        result = self._collection.get(
            include=["metadatas"]
        )

        metadatas = result.get(
            "metadatas",
            [],
        )

        topics = set()
        sources = set()
        bonus_topics_present = False

        for metadata in metadatas:
            topic = metadata.get(
                "topic"
            )

            if topic:
                topics.add(topic)

            source = metadata.get(
                "source"
            )

            if source:
                sources.add(source)

            if metadata.get(
                "is_bonus",
                False,
            ):
                bonus_topics_present = True

        return {
            "total_chunks": len(
                metadatas
            ),
            "topics": sorted(
                topics
            ),
            "sources": sorted(
                sources
            ),
            "bonus_topics_present": (
                bonus_topics_present
            ),
        }

    # ------------------------------------------------------------------
    # Delete document
    # ------------------------------------------------------------------

    def delete_document(
        self,
        source: str,
    ) -> int:
        """
        Delete all chunks belonging to a document.

        Returns the number of deleted chunks.
        """

        result = self._collection.get(
            where={
                "source": source
            }
        )

        ids = result.get(
            "ids",
            [],
        )

        if not ids:
            return 0

        self._collection.delete(
            ids=ids
        )

        logger.info(
            "Deleted {} chunks from {}",
            len(ids),
            source,
        )

        return len(ids)