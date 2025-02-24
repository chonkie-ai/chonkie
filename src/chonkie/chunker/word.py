"""Word-based chunker."""

import re
import warnings
from typing import Any, Callable, List, Literal, Tuple, Union

from chonkie.types import Chunk

from .base import BaseChunker


class WordChunker(BaseChunker):

    """Chunker that splits text into overlapping chunks based on words.

    Args:
        tokenizer: The tokenizer instance to use for encoding/decoding
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Maximum number of tokens to overlap between chunks
        return_type: Whether to return chunks or texts

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size

    """

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable, Any] = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        return_type: Literal["chunks", "texts"] = "chunks",
    ):
        """Initialize the WordChunker with configuration parameters.

        Args:
            tokenizer_or_token_counter: The tokenizer or token counter to use for encoding/decoding
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Maximum number of tokens to overlap between chunks
            return_type: Whether to return chunks or texts

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size or invalid return_type

        """
        super().__init__(tokenizer_or_token_counter=tokenizer_or_token_counter)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if return_type not in ["chunks", "texts"]:
            raise ValueError("Invalid return_type. Must be either 'chunks' or 'texts'.")

        # Add chunk_overlap deprecation warning
        if chunk_overlap > 0:
            warnings.warn(
                "chunk_overlap is getting deprecated in v0.6.0. " +
                "🦛 Chonkie advises you to use OverlapRefinery instead which is more flexible and powerful!",
                DeprecationWarning,
            )

        # Assign the values if they make sense
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.return_type = return_type

    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words while preserving whitespace."""
        split_points = [match.end() for match in re.finditer(r"(\s*\S+)", text)]
        words = []
        prev = 0

        for point in split_points:
            words.append(text[prev:point])
            prev = point

        if prev < len(text):
            words.append(text[prev:])

        return words

    def _create_chunk(
        self,
        words: List[str],
        text: str,
        token_count: int,
        current_index: int = 0,
    ) -> Tuple[Chunk, int]:
        """Create a chunk from a list of words.

        Args:
            words: List of words to create chunk from
            text: The original text
            token_count: Number of tokens in the chunk
            current_index: The index of the first token in the chunk

        Returns:
            Tuple of (Chunk object, number of tokens in chunk)

        """
        chunk_text = "".join(words)
        start_index = text.find(chunk_text, current_index)
        return Chunk(
            text=chunk_text,
            start_index=start_index,
            end_index=start_index + len(chunk_text),
            token_count=token_count,
        )

    def _get_word_list_token_counts(self, words: List[str]) -> List[int]:
        """Get the number of tokens for each word in a list.

        Args:
            words: List of words

        Returns:
            List of token counts for each word

        """
        words = [
            word for word in words if word != ""
        ]  # Add space in the beginning because tokenizers usually split that differently
        return [self.tokenizer.count_tokens(word) for word in words]

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into overlapping chunks based on words while respecting token limits.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        words = self._split_into_words(text)
        lengths = self._get_word_list_token_counts(words)
        chunks = []

        # Saving the current chunk
        current_chunk = []
        current_chunk_length = 0

        current_index = 0

        for i, (word, length) in enumerate(zip(words, lengths)):
            if current_chunk_length + length <= self.chunk_size:
                current_chunk.append(word)
                current_chunk_length += length
            else:
                if self.return_type == "chunks":
                    chunk = self._create_chunk(
                        current_chunk,
                        text,
                        current_chunk_length,
                        current_index,
                    )
                    chunks.append(chunk)
                elif self.return_type == "texts":
                    chunks.append("".join(current_chunk))

                # update the current_chunk and previous chunk
                previous_chunk_length = current_chunk_length
                current_index = chunk.end_index

                overlap = []
                overlap_length = 0
                # calculate the overlap from the current chunk in reverse
                for j in range(0, previous_chunk_length):
                    cwi = i - 1 - j
                    oword = words[cwi]
                    olength = lengths[cwi]
                    if overlap_length + olength <= self.chunk_overlap:
                        overlap.append(oword)
                        overlap_length += olength
                    else:
                        break

                current_chunk = [w for w in reversed(overlap)]
                current_chunk_length = overlap_length

                current_chunk.append(word)
                current_chunk_length += length

        # Add the final chunk if it has any words
        if current_chunk:
            if self.return_type == "chunks":
                chunk = self._create_chunk(current_chunk, text, current_chunk_length)
                chunks.append(chunk)
            elif self.return_type == "texts":
                chunks.append("".join(current_chunk))
        return chunks

    def __repr__(self) -> str:
        """Return a string representation of the WordChunker."""
        return (
            f"WordChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"return_type={self.return_type})"
        )
