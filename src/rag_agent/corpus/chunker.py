"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and overlap.
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.
        """

        # Validate that the file exists.
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        # Make sure the path is actually a file.
        if not file_path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        # Normalize file extension.
        suffix = file_path.suffix.lower()

        # Route to the appropriate chunking method.
        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(
                file_path,
                chunk_size,
                chunk_overlap,
            )

        elif suffix in {".md", ".markdown"}:
            raw_chunks = self._chunk_markdown(
                file_path,
                chunk_size,
                chunk_overlap,
            )

        else:
            raise ValueError(
                f"Unsupported file type: {suffix}"
            )

        # Infer metadata from filename and apply overrides.
        base_metadata = self._infer_metadata(
            file_path,
            metadata_overrides,
        )

        document_chunks = []

        for raw_chunk in raw_chunks:
            chunk_text = raw_chunk.get(
                "text",
                "",
            ).strip()

            if not chunk_text:
                continue

            # Generate deterministic chunk ID.
            chunk_id = (
                VectorStoreManager.generate_chunk_id(
                    str(file_path),
                    chunk_text,
                )
            )

            # Create DocumentChunk object.
            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=base_metadata,
                )
            )

        logger.info(
            "Chunked {} into {} chunks",
            file_path.name,
            len(document_chunks),
        )

        return document_chunks

    # -----------------------------------------------------------------------
    # Multiple Files
    # -----------------------------------------------------------------------

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.
        """

        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = self.chunk_file(
                    file_path,
                    metadata_overrides=metadata_overrides,
                )

                all_chunks.extend(chunks)

            except Exception as exc:
                logger.exception(
                    "Failed to chunk file {}",
                    file_path,
                )

        logger.info(
            "Chunked {} files into {} total chunks",
            len(file_paths),
            len(all_chunks),
        )

        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.
        """

        loader = PyPDFLoader(
            str(file_path)
        )

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        split_documents = splitter.split_documents(
            documents
        )

        chunks = []

        for document in split_documents:
            page = document.metadata.get(
                "page"
            )

            chunks.append(
                {
                    "text": document.page_content,
                    "page": page,
                }
            )

        logger.info(
            "PDF {} produced {} chunks",
            file_path.name,
            len(chunks),
        )

        return chunks

    # -----------------------------------------------------------------------
    # Markdown Chunking
    # -----------------------------------------------------------------------

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure, then RecursiveCharacterTextSplitter for
        oversized sections.
        """

        markdown_text = file_path.read_text(
            encoding="utf-8"
        )

        # Split according to Markdown headers first.
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        header_documents = (
            header_splitter.split_text(
                markdown_text
            )
        )

        # Further split sections that are too large.
        recursive_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

        split_documents = (
            recursive_splitter.split_documents(
                header_documents
            )
        )

        chunks = []

        for document in split_documents:
            chunks.append(
                {
                    "text": document.page_content,
                    "header": document.metadata,
                }
            )

        logger.info(
            "Markdown {} produced {} chunks",
            file_path.name,
            len(chunks),
        )

        return chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions
        and apply explicit overrides.

        Expected filename format:

            <topic>_<difficulty>.md

        Examples:

            lstm_intermediate.md
            cnn_beginner.md
            alexnet_advanced.pdf
        """

        stem = file_path.stem

        # Supported difficulty levels.
        difficulty_levels = {
            "beginner",
            "intermediate",
            "advanced",
        }

        parts = stem.split("_")

        # Detect difficulty from the final filename component.
        if (
            len(parts) >= 2
            and parts[-1].lower()
            in difficulty_levels
        ):
            topic = "_".join(
                parts[:-1]
            )

            difficulty = (
                parts[-1].lower()
            )

        else:
            topic = stem
            difficulty = "intermediate"

        # Convert underscores to spaces.
        topic = topic.replace(
            "_",
            " ",
        ).strip()

        # Bonus topics.
        bonus_topics = {
            "som",
            "boltzmannmachine",
            "boltzmann machine",
            "gan",
        }

        is_bonus = (
            topic.lower()
            in bonus_topics
        )

        metadata = {
            "source": file_path.name,
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept_explanation",
            "related_topics": [],
            "is_bonus": is_bonus,
        }

        # Apply explicit metadata overrides.
        if overrides:
            metadata.update(
                overrides
            )

        return ChunkMetadata(
            **metadata
        )