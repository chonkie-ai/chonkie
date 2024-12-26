"""Token-based chunking."""

from itertools import accumulate
from typing import Any, Generator, List, Tuple, Union

from chonkie.types import Chunk

from .base import BaseChunker


class TokenChunker(BaseChunker):
    """Chunker that splits text into chunks of a specified token size.

    Args:
        tokenizer: The tokenizer instance to use for encoding/decoding
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks

    """

    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: Union[int, float] = 128,
    ) -> None:
        """Initialize the TokenChunker with configuration parameters.

        Args:
            tokenizer: The tokenizer instance to use for encoding/decoding
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size

        """
        super().__init__(tokenizer)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if isinstance(chunk_overlap, int) and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if isinstance(chunk_overlap, float) and chunk_overlap >= 1:
            raise ValueError("chunk_overlap must be less than 1")

        self.chunk_size = chunk_size
        self.chunk_overlap = (
            chunk_overlap
            if isinstance(chunk_overlap, int)
            else int(chunk_overlap * chunk_size)
        )
    
    def _create_chunks(
        self,
        chunk_texts: List[str],
        token_counts: List[int],
        decoded_text: str,
    ) -> List[Chunk]:
        """Create chunks from a list of texts."""
        # package everything as Chunk objects and send out the result
        chunks = []
        current_index = 0
        for chunk_text, token_count in zip(chunk_texts, token_counts):
            start_index = decoded_text.find(
                chunk_text, current_index
            )  # Find needs to be run every single time because of unknown overlap length
            end_index = start_index + len(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count,
                )
            )
            current_index = end_index
        return chunks

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into overlapping chunks of specified token size.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        # Encode full text
        text_tokens = self._encode(text)

        # We decode the text because the tokenizer might result in a different output than text
        decoded_text = self._decode(text_tokens)

        # Calculate chunk positions
        token_groups = [
            text_tokens[
                start_index : min(start_index + self.chunk_size, len(text_tokens))
            ]
            for start_index in range(
                0, len(text_tokens), self.chunk_size - self.chunk_overlap
            )
        ]
        token_counts = [
            len(toks) for toks in token_groups
        ]  # get the token counts; it's prolly chunk_size, but len doesn't take too long

        chunk_texts = self._decode_batch(
            token_groups
        )  # decrease the time by decoding in one go (?)

        chunks = self._create_chunks(chunk_texts, token_counts, decoded_text)

        return chunks

    def _chunk_generator(
        self, tokens: List[int]
    ) -> Generator[Tuple[List[int], int, int], None, None]:
        """Generate chunks from a list of tokens."""
        stride = self.chunk_size - self.chunk_overlap
        for start in range(0, len(tokens), stride):
            end = min(start + self.chunk_size, len(tokens))
            yield tokens[start:end], start, end
            if end == len(tokens):
                break

    def _process_batch(self,
                       chunks: List[Tuple[List[int], int, int]],
                       full_text: str) -> List[Chunk]:
        """Process a batch of chunks."""
        token_lists = [tokens for tokens, _, _ in chunks]
        texts = self._decode_batch(token_lists)

        index_pairs = []
        current_index = 0
        for text in texts:
            start_index = full_text.find(text, current_index)
            end_index = start_index + len(text)
            index_pairs.append((start_index, end_index))
            current_index = end_index
            
        return [
            Chunk(text=text, start_index=start, end_index=end, token_count=len(tokens))
            for text, (start, end), tokens in zip(texts, index_pairs, token_lists)
        ]

    def _process_text_batch(self, texts: List[str]) -> List[List[Chunk]]:
        """Process a batch of texts."""
        tokens_list = self._encode_batch(texts)
        decoded_texts = self._decode_batch(tokens_list)
        result = []

        for tokens, text in zip(tokens_list, decoded_texts):
            if not tokens:
                result.append([])
                continue

            chunks = []
            chunk_batch = []

            for chunk_data in self._chunk_generator(tokens):
                chunk_batch.append(chunk_data)

            chunks.extend(self._process_batch(chunk_batch, text))
            result.append(chunks)

        return result

    def chunk_batch(
        self, texts: List[str], batch_size: Union[int, None] = None
    ) -> List[List[Chunk]]:
        """Split a batch of texts into their respective chunks.

        Args:
            texts: List of input texts to be chunked
            batch_size: Number of texts to process in a single batch

        Returns:
            List of lists of Chunk objects containing the chunked text and metadata

        """
        if batch_size is not None:
            chunks = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : min(i + batch_size, len(texts))]
                chunks.extend(self._process_text_batch(batch_texts))
            return chunks
        else:
            return self._process_text_batch(texts)

    def __repr__(self) -> str:
        """Return a string representation of the TokenChunker."""
        return (
            f"TokenChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
