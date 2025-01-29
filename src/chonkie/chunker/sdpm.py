"""Semantic Double Pass Merge chunking using sentence embeddings."""

from typing import Any, List, Union, Literal

from chonkie.types import SemanticChunk, Sentence

from .semantic import SemanticChunker


class SDPMChunker(SemanticChunker):
    """Chunker implementation using the Semantic Document Partitioning Method (SDPM).

    The SDPM approach involves three main steps:
    1. Grouping sentences by semantic similarity (Same as SemanticChunker)
    2. Merging similar groups with a skip window
    3. Splitting the merged groups into size-appropriate chunks

    Args:
        embedding_model: Sentence embedding model to use
        mode: Mode for grouping sentences, either "cumulative" or "window"
        threshold: Threshold for semantic similarity (0-1) or percentile (1-100), defaults to "auto"
        chunk_size: Maximum token count for a chunk
        similarity_window: Number of sentences to consider for similarity threshold calculation
        min_sentences: Minimum number of sentences per chunk
        min_chunk_size: Minimum number of tokens per sentence
        min_characters_per_sentence: Minimum number of characters per sentence
        threshold_step: Step size for similarity threshold calculation
        delim: Delimiters to split sentences on
        skip_window: Number of chunks to skip when looking for similarities
        return_type: Whether to return chunks or texts

    """

    def __init__(
        self,
        embedding_model: Union[str, Any] = "minishlab/potion-base-8M",
        mode: str = "window",
        threshold: Union[str, float, int] = "auto",
        chunk_size: int = 512,
        similarity_window: int = 1,
        min_sentences: int = 1,
        min_chunk_size: int = 2,
        min_characters_per_sentence: int = 12,
        threshold_step: float = 0.01,
        delim: Union[str, List[str]] = [".", "!", "?", "\n"],
        skip_window: int = 1,
        return_type: Literal["chunks", "texts"] = "chunks",
        **kwargs
    ):
        """Initialize the SDPMChunker.

        Args:
            embedding_model: Sentence embedding model to use
            mode: Mode for grouping sentences, either "cumulative" or "window"
            threshold: Threshold for semantic similarity (0-1) or percentile (1-100), defaults to "auto"
            chunk_size: Maximum token count for a chunk
            similarity_window: Number of sentences to consider for similarity threshold calculation
            min_sentences: Minimum number of sentences per chunk
            min_chunk_size: Minimum number of tokens per sentence
            min_characters_per_sentence: Minimum number of characters per sentence
            threshold_step: Step size for similarity threshold calculation
            delim: Delimiters to split sentences on
            skip_window: Number of chunks to skip when looking for similarities
            return_type: Whether to return chunks or texts
            **kwargs: Additional keyword arguments

        """
        super().__init__(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            mode=mode,
            threshold=threshold,
            similarity_window=similarity_window,
            min_sentences=min_sentences,
            min_chunk_size=min_chunk_size,
            min_characters_per_sentence=min_characters_per_sentence,
            threshold_step=threshold_step,
            delim=delim,
            return_type=return_type,
            **kwargs
        )
        self.skip_window = skip_window

        # Remove the multiprocessing flag from the base class
        self._use_multiprocessing = False

    def _merge_groups(self, groups: List[List[Sentence]]) -> List[Sentence]:
        """Merge the groups together."""
        merged_group = []
        for group in groups:
            merged_group.extend(group)
        return merged_group

    def _skip_and_merge(
        self, groups: List[List[Sentence]], similarity_threshold: float
    ) -> List[List[Sentence]]:
        """Merge similar groups considering skip window."""
        if len(groups) <= 1:
            return groups

        merged_groups = []
        embeddings = [self._compute_group_embedding(group) for group in groups]

        while groups:
            if len(groups) == 1:
                merged_groups.append(groups[0])
                break

            # Calculate skip index ensuring it's valid
            skip_index = min(self.skip_window + 1, len(groups) - 1)

            # Compare current group with skipped group
            similarity = self._get_semantic_similarity(
                embeddings[0], embeddings[skip_index]
            )

            if similarity >= similarity_threshold:
                # Merge groups from 0 to skip_index (inclusive)
                merged = self._merge_groups(groups[: skip_index + 1])

                # Remove the merged groups
                for _ in range(skip_index + 1):
                    groups.pop(0)
                    embeddings.pop(0)

                # Add merged group back at the start
                groups.insert(0, merged)
                embeddings.insert(0, self._compute_group_embedding(merged))
            else:
                # No merge possible, move first group to results
                merged_groups.append(groups.pop(0))
                embeddings.pop(0)

        return merged_groups

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into chunks using the SPDM approach.

        Args:
            text: Input text to be chunked

        Returns:
            List of SemanticChunk objects

        """
        if not text.strip():
            return []

        # Prepare sentences with precomputed information
        sentences = self._prepare_sentences(text)
        if len(sentences) <= self.min_sentences:
            return [self._create_chunk(sentences)]

        # Calculate similarity threshold
        self.similarity_threshold = self._calculate_similarity_threshold(sentences)

        # First pass: Group sentences by semantic similarity
        initial_groups = self._group_sentences(sentences)

        # Second pass: Merge similar groups with skip window
        merged_groups = self._skip_and_merge(initial_groups, self.similarity_threshold)

        # Final pass: Split into size-appropriate chunks
        chunks = self._split_chunks(merged_groups)

        return chunks

    def __repr__(self) -> str:
        """Return a string representation of the SDPMChunker."""
        return (
            f"SPDMChunker(embedding_model={self.embedding_model}, "
            f"mode={self.mode}, "
            f"threshold={self.threshold}, "
            f"chunk_size={self.chunk_size}, "
            f"similarity_window={self.similarity_window}, "
            f"min_sentences={self.min_sentences}, "
            f"min_chunk_size={self.min_chunk_size}, "
            f"min_characters_per_sentence={self.min_characters_per_sentence}, "
            f"threshold_step={self.threshold_step}, "
            f"skip_window={self.skip_window})"
        )
