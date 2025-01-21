"""Refinery class which adds overlap as context to chunks."""

from typing import Any, List, Optional, Tuple

from chonkie.refinery.base import BaseRefinery
from chonkie.types import Chunk, Context, SemanticChunk, SentenceChunk, RecursiveRules

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
        min_tokens: Optional[int] = None,  # New parameter
        tokenizer: Any = None,
        rules: Optional[RecursiveRules] = None,  # New parameter
        method: str = "static",  # New parameter
        mode: str = "suffix",
        merge_context: bool = True,
        inplace: bool = True,
        approximate: bool = True,
    ) -> None:
        """Initialize the OverlapRefinery class.

        Args:
            context_size: Number of tokens to include in context
            min_tokens: Minimum tokens required (only used with recursive method)
            tokenizer: Optional tokenizer for exact token counting
            rules: RecursiveRules for context boundary detection (only used with recursive method)
            method: Whether to use "static" or "recursive" context extraction
            mode: Whether to add context to prefix or suffix
            merge_context: Whether to merge context with chunk text
            inplace: Whether to update chunks in place
            approximate: Whether to use approximate token counting
        """
        super().__init__(context_size)
        
        # Validate method
        if method not in ["static", "recursive"]:
            raise ValueError("method must be either 'static' or 'recursive'")
            
        # Validate recursive-specific parameters
        if method == "recursive":
            if min_tokens is None:
                raise ValueError("min_tokens must be specified when using recursive method")
            if min_tokens > context_size:
                raise ValueError("min_tokens cannot be greater than context_size")
                
        self.method = method
        self.min_tokens = min_tokens
        self.rules = rules if rules else RecursiveRules()
        self.merge_context = merge_context
        self.inplace = inplace
        self.mode = mode

        # Set up tokenizer and counting method
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.approximate = approximate
        else:
            self.approximate = True
            
        # Average number of characters per token (for approximation)
        self._AVG_CHAR_PER_TOKEN = 7
        
        # Create hierarchical refinery if needed
        if method == "recursive":
            self._hierarchical = _HierarchicalRefinery(
                context_size=context_size,
                min_tokens=min_tokens,
                tokenizer=tokenizer,
                rules=rules,
                mode=mode,
                merge_context=merge_context,
                inplace=inplace,
                approximate=approximate
            )

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
        tokens = self.tokenizer.encode(text_portion) #TODO: should be self._encode; need a unified tokenizer interface
        context_tokens = min(self.context_size, len(tokens))
        context_tokens_ids = tokens[-context_tokens:]
        context_text = self.tokenizer.decode(context_tokens_ids) #TODO: should be self._decode; need a unified tokenizer interface

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
        tokens = self.tokenizer.encode(text_portion)
        context_tokens = min(self.context_size, len(tokens))
        context_tokens_ids = tokens[:context_tokens]
        context_text = self.tokenizer.decode(context_tokens_ids)

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
                        self.tokenizer.encode(refined_chunks[i].text)
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
                        self.tokenizer.encode(refined_chunks[i].text)
                    )
                else:
                    # Otherwise use approximate by adding context tokens
                    refined_chunks[i].token_count = (
                        refined_chunks[i].token_count + context.token_count
                    )

        return refined_chunks

    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by adding appropriate context."""
        if self.method == "recursive":
            return self._hierarchical.refine(chunks)
        
        # Original static overlap logic
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

class _HierarchicalRefinery(OverlapRefinery):
    """Refinery that uses recursive rules to add hierarchical context.
    
    This refinery extends OverlapRefinery to add rule-based context selection.
    It processes rules in sequence to find optimal context boundaries while 
    maintaining context size between specified minimum and maximum limits.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the HierarchicalRefinery.

        Args:
            context_size: Maximum number of tokens for context
            min_tokens: Minimum tokens required for context
            tokenizer: Tokenizer for exact token counting
            rules: RecursiveRules for context boundary detection
            mode: Whether to add context to prefix or suffix
            merge_context: Whether to merge context with chunk text
            inplace: Whether to modify chunks in place
            approximate: Whether to use approximate token counting
        """

        # Extract min_tokens before calling super()
        min_tokens = kwargs.get('min_tokens')
        context_size = kwargs.get('context_size')
        rules = kwargs.get('rules')
        
        if min_tokens is None:
            raise ValueError("min_tokens must be specified for HierarchicalRefinery")
            
        if min_tokens > context_size:
            raise ValueError("min_tokens cannot be greater than context_size")
        
        super().__init__(**kwargs)

        self.min_tokens = min_tokens
        self.context_size = context_size
        self.rules = rules if rules else RecursiveRules()
        
        # Set up token counting method based on tokenizer availability
        if hasattr(self, "tokenizer") and not self.approximate:
            self._count_tokens = self._exact_token_count
        else:
            self._count_tokens = self._approximate_token_count
                
    def _exact_token_count(self, text: str) -> int:
        """Count tokens exactly using tokenizer."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def _approximate_token_count(self, text: str) -> int:
        """Approximate token count based on text length."""
        if not text:
            return 0
        return len(text) // self._AVG_CHAR_PER_TOKEN

    def _find_primary_boundary_context(self, text: str) -> Optional[Tuple[str, int]]:
        """Find the smallest chunk from the end using primary rule that meets min_tokens.
        
        Uses the first level of recursive rules to find natural boundaries,
        accumulating from the end until reaching minimum token requirement.
        
        Args:
            text: Text to search for boundaries
            
        Returns:
            Tuple of (context_text, start_position) if found, None otherwise
        """
        if not self.rules or len(self.rules) == 0:
            return None
            
        # Get primary rule (first level)
        primary_rule = self.rules[0]
        if not primary_rule.delimiters:
            return None
            
        # Try each primary delimiter
        for delimiter in primary_rule.delimiters:
            parts = text.split(delimiter)
            if len(parts) <= 1:
                continue
                
            # Build context from the end
            current_text = ""
            accumulated_parts = []
            
            # Work backwards through parts
            for part in reversed(parts):
                if not part:  # Skip empty parts
                    continue
                    
                # Test current chunk plus delimiter
                test_text = part + delimiter + current_text
                token_count = self._count_tokens(test_text)
                
                if token_count >= self.min_tokens:
                    # Found minimal valid chunk
                    if accumulated_parts:
                        final_text = part + delimiter + delimiter.join(accumulated_parts)
                    else:
                        final_text = test_text
                        
                    # Find where this chunk starts in original text
                    try:
                        start_pos = text.rindex(final_text)
                        return final_text, start_pos
                    except ValueError:
                        continue
                        
                # Keep accumulating if under minimum
                accumulated_parts.insert(0, part)
                current_text = test_text
                
        return None
        
    def _get_hierarchical_context(self, chunk: Chunk) -> Optional[Context]:
        """Get context using hierarchical rules.
        
        First tries to find appropriate boundaries using primary rule,
        then falls back to secondary rules if needed.
        
        Args:
            chunk: Chunk to extract context from
            
        Returns:
            Context object with appropriate boundaries
        """
        if not chunk.text:
            return None
            
        # First try primary boundaries
        primary_result = self._find_primary_boundary_context(chunk.text)
        
        if primary_result:
            context_text, start_pos = primary_result
            return Context(
                text=context_text,
                token_count=self._count_tokens(context_text),
                start_index=chunk.start_index + start_pos,
                end_index=chunk.end_index
            )
            
        # Fallback to trying other rules if primary boundary didn't work
        for level_index in range(1, len(self.rules)):
            rule = self.rules[level_index]
            if not rule.delimiters and not rule.whitespace:
                continue
                
            if rule.delimiters:
                for delimiter in rule.delimiters:
                    pos = chunk.text.rfind(delimiter)
                    while pos >= 0:
                        candidate_text = chunk.text[pos + len(delimiter):]
                        token_count = self._count_tokens(candidate_text)
                        
                        if self.min_tokens <= token_count <= self.context_size:
                            return Context(
                                text=candidate_text,
                                token_count=token_count,
                                start_index=chunk.start_index + pos + len(delimiter),
                                end_index=chunk.end_index
                            )
                        pos = chunk.text.rfind(delimiter, 0, pos)
                        
            elif rule.whitespace:
                words = chunk.text.split()
                current_text = ""
                for word in reversed(words):
                    if current_text:
                        test_text = word + " " + current_text
                    else:
                        test_text = word
                    token_count = self._count_tokens(test_text)
                    
                    if token_count >= self.min_tokens:
                        try:
                            start_pos = chunk.text.rindex(test_text)
                            return Context(
                                text=test_text,
                                token_count=token_count,
                                start_index=chunk.start_index + start_pos,
                                end_index=chunk.end_index
                            )
                        except ValueError:
                            continue
                            
                    current_text = test_text
        
        # Last resort: use the whole chunk
        return Context(
            text=chunk.text,
            token_count=self._count_tokens(chunk.text),
            start_index=chunk.start_index,
            end_index=chunk.end_index
        )

    def _refine_prefix(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by adding hierarchical context to prefix.
        
        Override of OverlapRefinery method to use hierarchical rules.
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
            
        # Process chunks after first
        for i in range(1, len(refined_chunks)):
            # Get hierarchical context from previous chunk
            context = self._get_hierarchical_context(chunks[i - 1])
            setattr(refined_chunks[i], "context", context)
            
            # Optionally update chunk text to include context
            if self.merge_context and context:
                refined_chunks[i].text = context.text + refined_chunks[i].text
                refined_chunks[i].start_index = context.start_index
                refined_chunks[i].token_count = (
                    refined_chunks[i].token_count + context.token_count
                )
                
        return refined_chunks
        
    def _find_forward_boundary_context(self, text: str) -> Optional[Tuple[str, int]]:
        """Find the smallest chunk from the start using primary rule that meets min_tokens.
        
        Uses the first level of recursive rules to find natural boundaries,
        accumulating from the start until reaching minimum token requirement.
        
        Args:
            text: Text to search for boundaries
            
        Returns:
            Tuple of (context_text, end_position) if found, None otherwise
        """
        if not self.rules or len(self.rules) == 0:
            return None
            
        # Get primary rule (first level)
        primary_rule = self.rules[0]
        if not primary_rule.delimiters:
            return None
            
        # Try each primary delimiter
        for delimiter in primary_rule.delimiters:
            parts = text.split(delimiter)
            if len(parts) <= 1:
                continue
                
            # Build context from the start
            current_text = ""
            accumulated_parts = []
            
            # Work forwards through parts
            for part in parts:
                if not part:  # Skip empty parts
                    continue
                    
                # Test current chunk plus delimiter
                test_text = current_text + delimiter + part if current_text else part
                token_count = self._count_tokens(test_text)
                
                if token_count >= self.min_tokens:
                    # Found minimal valid chunk
                    if accumulated_parts:
                        final_text = delimiter.join(accumulated_parts) + delimiter + part
                    else:
                        final_text = test_text
                        
                    # Find where this chunk ends in original text
                    try:
                        start_pos = text.index(final_text)
                        return final_text, start_pos + len(final_text)
                    except ValueError:
                        continue
                        
                # Keep accumulating if under minimum
                accumulated_parts.append(part)
                current_text = test_text
                
        return None

    def _get_forward_hierarchical_context(self, chunk: Chunk) -> Optional[Context]:
        """Get context using hierarchical rules, working forwards.
        
        First tries to find appropriate boundaries using primary rule,
        then falls back to secondary rules if needed.
        
        Args:
            chunk: Chunk to extract context from
            
        Returns:
            Context object with appropriate boundaries
        """
        if not chunk.text:
            return None
            
        # First try primary boundaries
        primary_result = self._find_forward_boundary_context(chunk.text)
        
        if primary_result:
            context_text, end_pos = primary_result
            return Context(
                text=context_text,
                token_count=self._count_tokens(context_text),
                start_index=chunk.start_index,
                end_index=chunk.start_index + end_pos
            )
            
        # Fallback to trying other rules if primary boundary didn't work
        for level_index in range(1, len(self.rules)):
            rule = self.rules[level_index]
            if not rule.delimiters and not rule.whitespace:
                continue
                
            if rule.delimiters:
                for delimiter in rule.delimiters:
                    pos = chunk.text.find(delimiter)
                    while pos >= 0:
                        candidate_text = chunk.text[:pos]
                        token_count = self._count_tokens(candidate_text)
                        
                        if self.min_tokens <= token_count <= self.context_size:
                            return Context(
                                text=candidate_text,
                                token_count=token_count,
                                start_index=chunk.start_index,
                                end_index=chunk.start_index + pos
                            )
                        pos = chunk.text.find(delimiter, pos + 1)
                        
            elif rule.whitespace:
                words = chunk.text.split()
                current_text = ""
                for word in words:
                    if current_text:
                        test_text = current_text + " " + word
                    else:
                        test_text = word
                    token_count = self._count_tokens(test_text)
                    
                    if token_count >= self.min_tokens:
                        try:
                            end_pos = chunk.text.index(test_text) + len(test_text)
                            return Context(
                                text=test_text,
                                token_count=token_count,
                                start_index=chunk.start_index,
                                end_index=chunk.start_index + end_pos
                            )
                        except ValueError:
                            continue
                            
                    current_text = test_text
        
        # Last resort: use the whole chunk
        return Context(
            text=chunk.text,
            token_count=self._count_tokens(chunk.text),
            start_index=chunk.start_index,
            end_index=chunk.end_index
        )

    def _refine_suffix(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by adding hierarchical context to suffix.
        
        Uses forward context search from the start of the next chunk.
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
            
        # Process chunks before last
        for i in range(len(refined_chunks) - 1):
            # Get hierarchical context from next chunk, working forwards
            context = self._get_forward_hierarchical_context(chunks[i + 1])
            setattr(refined_chunks[i], "context", context)
            
            # Optionally update chunk text to include context
            if self.merge_context and context:
                refined_chunks[i].text = refined_chunks[i].text + context.text
                refined_chunks[i].end_index = context.end_index
                refined_chunks[i].token_count = (
                    refined_chunks[i].token_count + context.token_count
                )
                
        return refined_chunks

    @classmethod
    def is_available(cls) -> bool:
        """Check if HierarchicalRefinery is available."""
        return True