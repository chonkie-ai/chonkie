"""Refinery class which adds overlap as context to chunks."""

from typing import Any, List, Optional

from chonkie.refinery.base import BaseRefinery
from chonkie.types import Chunk, Context, SemanticChunk, SentenceChunk


class OverlapRefinery(BaseRefinery):
    """Refinery class which adds overlap as context to chunks.

    This refinery provides two methods for calculating overlap:
    1. Exact: Uses a tokenizer to precisely determine token boundaries
    2. Approximate: Estimates tokens based on text length ratios

    It can handle different types of chunks (basic Chunks, SentenceChunks,
    and SemanticChunks) and can optionally update the chunk text to include
    the overlap content.
    """

    def __init__(
        self,
        context_size: int = 128,
        tokenizer: Any = None,
        mode: str = "suffix",
        merge_context: bool = True,
        inplace: bool = True,
        approximate: bool = True,
    ) -> None:
        """Initialize the OverlapRefinery class.

        Args:
            context_size: Number of tokens to include in context
            tokenizer: Optional tokenizer for exact token counting
            merge_context: Whether to merge context with chunk text
            inplace: Whether to update chunks in place
            approximate: Whether to use approximate token counting
            mode: Whether to add context to the prefix or suffix

        """
        super().__init__(context_size)
        self.merge_context = merge_context
        self.inplace = inplace
        self.mode = mode

        # If tokenizer provided, we can do exact token counting
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self._tokenizer_backend = self._get_tokenizer_backend()
            self.approximate = approximate
        else:
            # Without tokenizer, must use approximate method
            self.approximate = True
        
        # Average number of characters per token
        self._AVG_CHAR_PER_TOKEN = 7
    
    def _get_tokenizer_backend(self) -> str:
        """Get the tokenizer backend."""
        if "tokenizers" in str(type(self.tokenizer)):
            return "tokenizers"
        elif "tiktoken" in str(type(self.tokenizer)):
            return "tiktoken"
        elif "transformers" in str(type(self.tokenizer)):
            return "transformers"
        else:
            raise ValueError(f"Unsupported tokenizer backend: {str(type(self.tokenizer))}")

    def _encode(self, text: str) -> List[int]:
        """Encode text using the tokenizer backend."""
        if self._tokenizer_backend == "tokenizers":
            return self.tokenizer.encode(text).ids
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode(text)
        elif self._tokenizer_backend == "transformers":
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            raise ValueError(f"Unsupported tokenizer backend: {self._tokenizer_backend}")

    def _decode(self, tokens: List[int]) -> str:
        """Decode tokens using the tokenizer backend."""
        if self._tokenizer_backend == "tokenizers":
            return self.tokenizer.decode(tokens)
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.decode(tokens)
        elif self._tokenizer_backend == "transformers":
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            raise ValueError(f"Unsupported tokenizer backend: {self._tokenizer_backend}")

    def _batch_encode(self, texts: List[str]) -> List[List[int]]:
        """Batch encode texts using the tokenizer backend."""
        if self._tokenizer_backend == "tokenizers":
            return [t.ids for t in self.tokenizer.encode_batch(texts)]
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode_batch(texts)
        elif self._tokenizer_backend == "transformers":
            return self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)["input_ids"]
        else:
            raise ValueError(f"Unsupported tokenizer backend: {self._tokenizer_backend}")

    def _batch_decode(self, tokens: List[List[int]]) -> List[str]:
        """Batch decode tokens using the tokenizer backend."""
        if self._tokenizer_backend == "tokenizers":
            return self.tokenizer.decode_batch(tokens)
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.decode_batch(tokens)
        elif self._tokenizer_backend == "transformers":
            return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        else:
            raise ValueError(f"Unsupported tokenizer backend: {self._tokenizer_backend}")     

    def _get_refined_chunks(
        self, chunks: List[Chunk], inplace: bool = True
    ) -> List[Chunk]:
        """Convert regular chunks to refined chunks with progressive memory cleanup.

        This method takes regular chunks and converts them to RefinedChunks one at a
        time. When inplace is True, it progressively removes chunks from the input
        list to minimize memory usage.

        The conversion preserves all relevant information from the original chunks,
        including sentences and embeddings if they exist. This allows us to maintain
        the full capabilities of semantic chunks while adding refinement features.

        Args:
            chunks: List of original chunks to convert
            inplace: Whether to modify the input list during conversion

        Returns:
            List of RefinedChunks without any context (context is added later)

        Example:
            For memory efficiency with large datasets:
            ```
            chunks = load_large_dataset()  # Many chunks
            refined = refinery._get_refined_chunks(chunks, inplace=True)
            # chunks is now empty, memory is freed
            ```

        """
        if not chunks:
            return []

        refined_chunks = []

        # Use enumerate to track position without modifying list during iteration
        for i in range(len(chunks)):
            if inplace:
                # Get and remove the first chunk
                chunk = chunks.pop(0)
            else:
                # Just get a reference if not modifying in place
                chunk = chunks[i]

            # Create refined version preserving appropriate attributes
            refined_chunk = SemanticChunk(
                text=chunk.text,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                token_count=chunk.token_count,
                # Preserve sentences and embeddings if they exist
                sentences=chunk.sentences
                if isinstance(chunk, (SentenceChunk, SemanticChunk))
                else None,
                embedding=chunk.embedding if isinstance(chunk, SemanticChunk) else None,
                context=None,  # Context is added later in the refinement process
            )

            refined_chunks.append(refined_chunk)

        if inplace:
            # Clear the input list to free memory
            chunks.clear()
            chunks += refined_chunks

        return refined_chunks

    def _prefix_overlap_token_exact(self, chunk: Chunk) -> Optional[Context]:
        """Calculate precise token-based overlap context using tokenizer.

        Takes a larger window of text from the chunk end, tokenizes it,
        and selects exactly context_size tokens worth of text.

        Args:
            chunk: Chunk to extract context from

        Returns:
            Context object with precise token boundaries, or None if no tokenizer

        """
        if not hasattr(self, "tokenizer"):
            return None

        # Take _AVG_CHAR_PER_TOKEN * context_size characters to ensure enough tokens
        char_window = min(len(chunk.text), int(self.context_size * self._AVG_CHAR_PER_TOKEN))
        text_portion = chunk.text[-char_window:]

        # Get exact token boundaries
        tokens = self._encode(text_portion) #TODO: should be self._encode; need a unified tokenizer interface
        context_tokens = min(self.context_size, len(tokens))
        context_tokens_ids = tokens[-context_tokens:]
        context_text = self._decode(context_tokens_ids) #TODO: should be self._decode; need a unified tokenizer interface

        # Find where context text starts in chunk
        try:
            context_start = chunk.text.rindex(context_text)
            start_index = chunk.start_index + context_start

            return Context(
                text=context_text,
                token_count=context_tokens,
                start_index=start_index,
                end_index=chunk.end_index,
            )
        except ValueError:
            # If context text can't be found (e.g., due to special tokens), fall back to approximate
            return self._prefix_overlap_token_approximate(chunk)

    def _suffix_overlap_token_exact(self, chunk: Chunk) -> Optional[Context]:
        """Calculate precise token-based overlap context using tokenizer.

        Takes a larger window of text from the chunk start, tokenizes it,
        and selects exactly context_size tokens worth of text.
        """
        if not hasattr(self, "tokenizer"):
            return None

        # Take _AVG_CHAR_PER_TOKEN * context_size characters to ensure enough tokens
        char_window = min(len(chunk.text), int(self.context_size * self._AVG_CHAR_PER_TOKEN))
        text_portion = chunk.text[:char_window]

        # Get exact token boundaries
        tokens = self._encode(text_portion)
        context_tokens = min(self.context_size, len(tokens))
        context_tokens_ids = tokens[:context_tokens]
        context_text = self._decode(context_tokens_ids)

        # Find where context text starts in chunk
        try:
            return Context(
                text=context_text,
                token_count=context_tokens,
                start_index=chunk.start_index,
                end_index=chunk.start_index + len(context_text),
            )
        except ValueError:
            # If context text can't be found (e.g., due to special tokens), fall back to approximate
            return self._suffix_overlap_token_approximate(chunk)

    def _prefix_overlap_token_approximate(self, chunk: Chunk) -> Optional[Context]:
        """Calculate approximate token-based overlap context.

        Estimates token positions based on character length ratios.

        Args:
            chunk: Chunk to extract context from

        Returns:
            Context object with estimated token boundaries

        """
        # Calculate desired context size
        context_tokens = min(self.context_size, chunk.token_count)

        # Estimate text length based on token ratio
        context_ratio = context_tokens / chunk.token_count
        char_length = int(len(chunk.text) * context_ratio)

        # Extract context text from end
        context_text = chunk.text[-char_length:]

        return Context(
            text=context_text,
            token_count=context_tokens,
            start_index=chunk.end_index - char_length,
            end_index=chunk.end_index,
        )

    def _suffix_overlap_token_approximate(self, chunk: Chunk) -> Optional[Context]:
        """Calculate approximate token-based overlap context.

        Estimates token positions based on character length ratios.
        """
        # Calculate desired context size
        context_tokens = min(self.context_size, chunk.token_count)

        # Estimate text length based on token ratio
        context_ratio = context_tokens / chunk.token_count
        char_length = int(len(chunk.text) * context_ratio)

        # Extract context text from end
        context_text = chunk.text[:char_length]

        return Context(
            text=context_text,
            token_count=context_tokens,
            start_index=chunk.start_index,
            end_index=chunk.start_index + char_length,
        )

    def _prefix_overlap_token(self, chunk: Chunk) -> Optional[Context]:
        """Choose between exact or approximate token overlap calculation.

        Args:
            chunk: Chunk to process

        Returns:
            Context object from either exact or approximate calculation

        """
        if self.approximate:
            return self._prefix_overlap_token_approximate(chunk)
        return self._prefix_overlap_token_exact(chunk)

    def _suffix_overlap_token(self, chunk: Chunk) -> Optional[Context]:
        """Choose between exact or approximate token overlap calculation.

        Args:
            chunk: Chunk to process

        Returns:
            Context object from either exact or approximate calculation

        """
        if self.approximate:
            return self._suffix_overlap_token_approximate(chunk)
        return self._suffix_overlap_token_exact(chunk)

    def _prefix_overlap_sentence(self, chunk: SentenceChunk) -> Optional[Context]:
        """Calculate overlap context based on sentences.

        Takes sentences from the end of the chunk up to context_size tokens.

        Args:
            chunk: SentenceChunk to process

        Returns:
            Context object containing complete sentences

        """
        if not chunk.sentences:
            return None

        context_sentences = []
        total_tokens = 0

        # Add sentences from the end until we hit context_size
        for sentence in reversed(chunk.sentences):
            if total_tokens + sentence.token_count <= self.context_size:
                context_sentences.insert(0, sentence)
                total_tokens += sentence.token_count
            else:
                break
        # If no sentences were added, add the last sentence
        if not context_sentences:
            context_sentences.append(chunk.sentences[-1])
            total_tokens = chunk.sentences[-1].token_count

        return Context(
            text="".join(s.text for s in context_sentences),
            token_count=total_tokens,
            start_index=context_sentences[0].start_index,
            end_index=context_sentences[-1].end_index,
        )

    def _suffix_overlap_sentence(self, chunk: SentenceChunk) -> Optional[Context]:
        """Calculate overlap context based on sentences from the start.

        Takes sentences from the start of the chunk up to context_size tokens.

        Args:
            chunk: SentenceChunk to process

        Returns:
            Context object containing complete sentences

        """
        if not chunk.sentences:
            return None

        context_sentences = []
        total_tokens = 0

        # Add sentences from the end until we hit context_size
        for sentence in chunk.sentences:
            if total_tokens + sentence.token_count <= self.context_size:
                context_sentences.append(sentence)
                total_tokens += sentence.token_count
            else:
                break
        # If no sentences were added, add the first sentence
        if not context_sentences:
            context_sentences.append(chunk.sentences[0])
            total_tokens = chunk.sentences[0].token_count

        return Context(
            text="".join(s.text for s in context_sentences),
            token_count=total_tokens,
            start_index=context_sentences[0].start_index,
            end_index=context_sentences[-1].end_index,
        )

    def _get_prefix_overlap_context(self, chunk: Chunk) -> Optional[Context]:
        """Get appropriate overlap context based on chunk type."""
        if isinstance(chunk, SemanticChunk) or isinstance(chunk, SentenceChunk):
            return self._prefix_overlap_sentence(chunk)
        elif isinstance(chunk, Chunk):
            return self._prefix_overlap_token(chunk)
        else:
            raise ValueError(f"Unsupported chunk type: {type(chunk)}")

    def _get_suffix_overlap_context(self, chunk: Chunk) -> Optional[Context]:
        """Get appropriate overlap context based on chunk type."""
        if isinstance(chunk, SemanticChunk) or isinstance(chunk, SentenceChunk):
            return self._suffix_overlap_sentence(chunk)
        elif isinstance(chunk, Chunk):
            return self._suffix_overlap_token(chunk)
        else:
            raise ValueError(f"Unsupported chunk type: {type(chunk)}")

    def _refine_prefix(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by adding overlap context to the prefix.

        For each chunk after the first, adds context from the previous chunk.
        Can optionally update the chunk text to include the context.

        Args:
            chunks: List of chunks to refine

        Returns:
            List of refined chunks with added context

        """
        if not chunks:
            return chunks

        # Validate chunk types
        if len(set(type(chunk) for chunk in chunks)) > 1:
            raise ValueError("All chunks must be of the same type")

        if not self.inplace:
            refined_chunks = [chunk.copy() for chunk in chunks]
        else:
            refined_chunks = chunks

        # Process remaining chunks
        for i in range(1, len(refined_chunks)):
            # Get context from previous chunk
            context = self._get_prefix_overlap_context(chunks[i - 1])
            setattr(refined_chunks[i], "context", context)

            # Optionally update chunk text to include context
            if self.merge_context and context:
                refined_chunks[i].text = context.text + refined_chunks[i].text
                refined_chunks[i].start_index = context.start_index
                # Update token count to include context and space
                # Calculate new token count
                if hasattr(self, "tokenizer") and not self.approximate:
                    # Use exact token count if we have a tokenizer
                    refined_chunks[i].token_count = len(
                        self._encode(refined_chunks[i].text)
                    )
                else:
                    # Otherwise use approximate by adding context tokens plus one for space
                    refined_chunks[i].token_count = (
                        refined_chunks[i].token_count + context.token_count
                    )

        return refined_chunks

    def _refine_suffix(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by adding overlap context to the suffix.

        For each chunk before the last, adds context from the next chunk.
        Can optionally update the chunk text to include the context.

        Args:
            chunks: List of chunks to refine

        Returns:
            List of refined chunks with added context

        """
        if not chunks:
            return chunks

        # Validate chunk types
        if len(set(type(chunk) for chunk in chunks)) > 1:
            raise ValueError("All chunks must be of the same type")

        if not self.inplace:
            refined_chunks = [chunk.copy() for chunk in chunks]
        else:
            refined_chunks = chunks

        # Process remaining chunks
        for i in range(len(refined_chunks) - 1):
            # Get context from next chunk
            context = self._get_suffix_overlap_context(chunks[i + 1])
            setattr(refined_chunks[i], "context", context)

            # Optionally update chunk text to include context
            if self.merge_context and context:
                refined_chunks[i].text = refined_chunks[i].text + context.text
                refined_chunks[i].end_index = context.end_index
                # Update token count to include context
                # Calculate new token count
                if hasattr(self, "tokenizer") and not self.approximate:
                    # Use exact token count if we have a tokenizer
                    refined_chunks[i].token_count = len(
                        self._encode(refined_chunks[i].text)
                    )
                else:
                    # Otherwise use approximate by adding context tokens
                    refined_chunks[i].token_count = (
                        refined_chunks[i].token_count + context.token_count
                    )

        return refined_chunks

    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by adding overlap context."""
        if self.mode == "prefix":
            return self._refine_prefix(chunks)
        elif self.mode == "suffix":
            return self._refine_suffix(chunks)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    @classmethod
    def is_available(cls) -> bool:
        """Check if the OverlapRefinery is available.

        Always returns True as this refinery has no external dependencies.
        """
        return True
