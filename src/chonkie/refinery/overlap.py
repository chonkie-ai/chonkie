"""Refinery class which adds overlap as context to chunks."""

from bisect import bisect_left
from itertools import accumulate
from typing import Any, List, Literal, Optional, Tuple

from chonkie.refinery.base import BaseRefinery
from chonkie.types import (
    Chunk,
    Context,
    RecursiveChunk,
    RecursiveLevel,
    RecursiveRules,
    SemanticChunk,
    SentenceChunk,
)


class OverlapRefinery(BaseRefinery):
    """Refinery class which adds overlap as context to chunks.

    It can handle different types of chunks (basic Chunks, SentenceChunks,
    SemanticChunks, RecursiveChunks) and can optionally update the chunk text to include
    the overlap content.
    """

    def __init__(
        self,
        context_size: int = 128,
        tokenizer: Any = None,
        mode: str = "auto",
        method: str = "suffix",
        merge_context: bool = True,
        inplace: bool = True,
        approximate: bool = True,
        rules: Optional[RecursiveRules] = RecursiveRules(),
    ) -> None:
        """Initialize the OverlapRefinery class.

        Args:
            context_size: Number of tokens to include in context
            tokenizer: Optional tokenizer for exact token counting
            merge_context: Whether to merge context with chunk text
            inplace: Whether to update chunks in place
            approximate: Whether to use approximate token counting
            mode: One of "auto", "token", "sentence", "recursive". This would make the
            refinery opperate on tokens, sentences or recursively
            method: One of "suffix", "prefix". This would make the refinery use suffix
            or prefix context
            rules: Rules for recursive context, if mode is "recursive"

        """
        super().__init__(context_size)
        
        # validate mode
        if mode not in ["auto", "token", "sentence", "recursive"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

        # validate method
        if method not in ["suffix", "prefix"]:
            raise ValueError(f"Invalid method: {method}")
        self.method = method

        # Set attributes
        self.merge_context = merge_context
        self.inplace = inplace
        self.rules = rules
        # If tokenizer provided, we can do exact token counting
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self._tokenizer_backend = self._get_tokenizer_backend()
            self.approximate = approximate
        else:
            # Without tokenizer, must use approximate method
            self.tokenizer = None
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
            return self.tokenizer.encode(text, add_special_tokens=False).ids
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

    def _encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Batch encode texts using the tokenizer backend."""
        if self._tokenizer_backend == "tokenizers":
            return [t.ids for t in self.tokenizer.encode_batch(texts, add_special_tokens=False)]
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode_batch(texts)
        elif self._tokenizer_backend == "transformers":
            return self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)["input_ids"]
        else:
            raise ValueError(f"Unsupported tokenizer backend: {self._tokenizer_backend}")

    def _decode_batch(self, tokens: List[List[int]]) -> List[str]:
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
        # If the chunk doesn't have the attribute sentences, raise an error
        if not hasattr(chunk, "sentences"):
            raise ValueError("Input chunk does not have attribute sentences. " +
                             "`mode=sentence` is currently only supported for SentenceChunk or SemanticChunk."+
                             f"Input chunk type: {type(chunk)}. If you truly wish for it to be supported, please open an issue on github.")
        
        # If the chunk has no sentences, return None
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
        # If the chunk doesn't have the attribute sentences, raise an error
        if not hasattr(chunk, "sentences"):
            raise ValueError("Input chunk does not have attribute sentences" +
                             "mode=sentence is only supported for SentenceChunk or SemanticChunk."+
                             f"Input chunk type: {type(chunk)}. If you truly wish for it to be supported, please open an issue on github.")
        
        # If the chunk has no sentences, return None
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
    
    def _split_text_at_rule(self, 
                            text: str,
                            rule: RecursiveLevel, 
                            sep: str = "🦛") -> List[str]:
        """Split the text at the current rule."""
        if rule.delimiters:
            for delimiter in rule.delimiters:
                text = text.replace(delimiter, delimiter + sep)

            # Split the text at the sep
            splits = [s for s in text.split(sep) if s != ""]

            # This is different from the recursive chunker in that
            # we don't merge splits that are too short in characters
            # because we don't worry about the min_characters_per_chunk

            return splits
        if rule.whitespace:
            return self._split_at_whitespace(text)
        else:
            return self._split_at_tokens(text)

    def _split_at_whitespace(self,
                             text: str) -> List[str]:
        """Split the text at whitespace."""
        return text.split(' ')
    
    def _split_at_tokens(self,
                         text: str) -> List[str]:
        """Split the text at tokens."""
        tokens = self._encode(text)

        # Split the tokens at the chunk size
        token_splits = [tokens[i:i+self.context_size] for i in range(0, len(tokens), self.context_size)]
        
        # Decode the tokens back to text
        splits = self._decode_batch(token_splits)
        return splits

    def _get_token_count(self, text: str) -> int:
        """Get the token count of a text."""
        estimate = max(1, len(text) // self._AVG_CHAR_PER_TOKEN)
        if estimate > self.context_size:
            return self.context_size + 1
        elif self.tokenizer is not None and not self.approximate:
            return len(self._encode(text))
        else:
            return estimate
    
    def _merge_splits(self, 
                     splits: List[str],
                     token_counts: List[int],
                     combine_with_whitespace: bool = False, 
                     reverse: bool = False, 
                     boundry: Literal["ahead", "behind"] = "ahead") -> Tuple[List[str], List[int]]:
        """Merge splits based on token counts."""
        # If the number of splits and token counts does not match, raise an error
        if len(splits) != len(token_counts):
            raise ValueError("The number of splits and token counts does not match.")
        
        # If the splits are larger than the context size, we can just return the splits
        if all(tc > self.context_size for tc in token_counts):
            return splits, token_counts
        
        # If reverse is True, we need to reverse the splits and token counts
        if reverse:
            splits = splits[::-1]
            token_counts = token_counts[::-1]

        # If the splits are too short, merge them
        merged = []

        # NOTE: When combining with or without whitespace, most tokenizers will not count the space as a token
        # so it makes no difference in the token counts
        cumulative_token_counts = list(accumulate([0] + token_counts, lambda x, y: x + y)) 

        # Iterate through the splits and merge them if they are too short

        current_index = 0
        merged_token_counts = []

        # Use bisect_left to find the index to merge at 
        while current_index < len(splits):
            current_token_count = cumulative_token_counts[current_index]
            required_token_count = current_token_count + self.context_size
            # print(current_index, current_token_count, required_token_count)

            # Find the index to merge at
            if boundry == "ahead":
                index = min(bisect_left(cumulative_token_counts, required_token_count, lo=current_index), len(splits)) # Does one larger than the required token count
            elif boundry == "behind":
                index = min(bisect_left(cumulative_token_counts, required_token_count, lo=current_index) - 1, len(splits)) # Does one smaller than the required token count
            else:
                raise ValueError(f"Invalid boundry: {boundry}")

            # If the index is the same as the current index, we need to merge the next split
            if index == current_index:
                index += 1

            # Merge the splits at the index
            if combine_with_whitespace:
                merged.append(" ".join(splits[current_index:index]))
            else:
                merged.append("".join(splits[current_index:index]))

            # Add the token count of the merged split
            merged_token_counts.append(cumulative_token_counts[min(index, len(splits))] - current_token_count)

            # Update the current index
            current_index = index

        if reverse:
            merged = merged[::-1]
            merged_token_counts = merged_token_counts[::-1]

        return merged, merged_token_counts

    def _prefix_overlap_recursive(self, 
                                  text: str,
                                  full_text: str,
                                  level: int = 0) -> Optional[Context]:
        """Recursively find the overlap context for a recursive chunk."""
        if not self.rules:
            return None
        
        # Split the text at the current rule
        rule = self.rules[level]
        splits = self._split_text_at_rule(text, rule)
        token_counts = [self._get_token_count(split) for split in splits]
        if not rule.whitespace:
            merged, merged_token_counts = self._merge_splits(splits, 
                                                             token_counts, 
                                                             combine_with_whitespace=False, 
                                                             reverse=True)
        else:
            merged, merged_token_counts = self._merge_splits(splits, 
                                                             token_counts, 
                                                             combine_with_whitespace=True, 
                                                             reverse=True)
        
        # Recursively find the overlap context for the merged splits if they are too long
        if merged_token_counts[-1] > self.context_size:
            return self._prefix_overlap_recursive(merged[-1], full_text, level + 1)
        else:
            start_index = full_text.find(merged[-1])
            end_index = start_index + len(merged[-1])
            print(merged[-1], merged_token_counts[-1], start_index, end_index)
            return Context(
                text=merged[-1],
                token_count=merged_token_counts[-1],
                start_index=start_index,
                end_index=end_index,
            )
    
    def _suffix_overlap_recursive(self, text: str, full_text: str, level: int = 0) -> Optional[Context]:
        """Get appropriate overlap context based on chunk type."""
        if not self.rules:
            return None
        
        # Split the text at the current rule
        rule = self.rules[level]
        splits = self._split_text_at_rule(text, rule)
        token_counts = [self._get_token_count(split) for split in splits]
        # If the rule has no whitespace, we can just merge the splits
        if not rule.whitespace:
            merged, merged_token_counts = self._merge_splits(splits, 
                                                             token_counts, 
                                                             combine_with_whitespace=False)
        else:
            merged, merged_token_counts = self._merge_splits(splits, 
                                                             token_counts, 
                                                             combine_with_whitespace=True)
        
        if merged_token_counts[0] > self.context_size:
            return self._suffix_overlap_recursive(merged[0], full_text, level + 1)
        else:
            start_index = full_text.find(merged[0])
            end_index = start_index + len(merged[0])
            return Context(
                text=merged[0],
                token_count=merged_token_counts[0],
                start_index=start_index,
                end_index=end_index,
            )
        
    def _get_prefix_overlap_context(self, chunk: Chunk) -> Optional[Context]:
        """Get appropriate overlap context based on chunk type."""
        # If the mode is auto, we need to check the type of the chunk
        if self.mode == "auto":
            if isinstance(chunk, RecursiveChunk):
                context = self._prefix_overlap_recursive(chunk.text, chunk.text)
                if context:
                    context.start_index += chunk.start_index
                    context.end_index += chunk.start_index
                    return context
            elif isinstance(chunk, SemanticChunk) or isinstance(chunk, SentenceChunk):
                return self._prefix_overlap_sentence(chunk)
            else:
                return self._prefix_overlap_token(chunk)
        # If the mode is recursive, we need to recursively find the overlap context
        elif self.mode == "recursive":
            context = self._prefix_overlap_recursive(chunk.text, chunk.text)
            if context:
                context.start_index += chunk.start_index
                context.end_index += chunk.start_index
                return context
        # If the mode is sentences, we need to find the overlap context based on sentences
        elif self.mode == "sentence":
            return self._prefix_overlap_sentence(chunk)
        # If the mode is tokens, we need to find the overlap context based on tokens
        elif self.mode == "token":
            return self._prefix_overlap_token(chunk)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _get_suffix_overlap_context(self, chunk: Chunk) -> Optional[Context]:
        """Get appropriate overlap context based on chunk type."""
        if self.mode == "auto":
            if isinstance(chunk, RecursiveChunk):
                context = self._suffix_overlap_recursive(chunk.text, chunk.text)
                if context:
                    context.start_index += chunk.start_index
                    context.end_index += chunk.start_index
                    return context
            elif isinstance(chunk, SemanticChunk) or isinstance(chunk, SentenceChunk):
                return self._suffix_overlap_sentence(chunk)
            else:
                return self._suffix_overlap_token(chunk)
        elif self.mode == "recursive":
            context = self._suffix_overlap_recursive(chunk.text, chunk.text)
            if context:
                context.start_index += chunk.start_index
                context.end_index += chunk.start_index
                return context
        elif self.mode == "sentence":
            return self._suffix_overlap_sentence(chunk)
        elif self.mode == "token":
            return self._suffix_overlap_token(chunk)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _refine_prefix(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by adding overlap context to the prefix.

        For each chunk after the first, adds context from the previous chunk.
        Can optionally update the chunk text to include the context.

        Args:
            chunks: List of chunks to refine

        Returns:
            List of refined chunks with added context

        """
        # If no chunks, return original chunks
        if not chunks:
            return chunks

        # Validate chunk types
        if len(set(type(chunk) for chunk in chunks)) > 1:
            raise ValueError("All chunks must be of the same type")

        # If not inplace, create a copy of the chunks
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
        if self.method == "prefix":
            return self._refine_prefix(chunks)
        elif self.method == "suffix":
            return self._refine_suffix(chunks)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    @classmethod
    def is_available(cls) -> bool:
        """Check if the OverlapRefinery is available.

        Always returns True as this refinery has no external dependencies.
        """
        return True