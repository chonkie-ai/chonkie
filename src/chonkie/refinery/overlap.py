"""Refinery class which adds overlap as context to chunks."""

from typing import Any, List, Optional, Tuple

from chonkie.refinery.base import BaseRefinery
from chonkie.types import Chunk, Context, SemanticChunk, SentenceChunk, RecursiveRules

class OverlapRefinery(BaseRefinery):
    """Refinery class which adds overlap as context to chunks.

    This refinery provides three methods for calculating overlap:
    1. Exact: Uses a tokenizer to precisely determine token boundaries
    2. Approximate: Estimates tokens based on text length ratios
    3. Recursive: Uses hierarchical rules to find natural boundaries in text

    It can handle different types of chunks (basic Chunks, SentenceChunks,
    and SemanticChunks) and can optionally update the chunk text to include
    the overlap content.
    """

    def __init__(
        self,
        context_size: int = 128,
        min_tokens: Optional[int] = None,
        tokenizer: Any = None,
        rules: Optional[RecursiveRules] = None,
        method: str = "static",
        mode: str = "suffix",
        merge_context: bool = True,
        inplace: bool = True,
        approximate: bool = True,
    ) -> None:
        """Initialize the OverlapRefinery class.
        
        Args:
            context_size: Maximum number of tokens to include in context
            min_tokens: Minimum number of tokens required for recursive method
            tokenizer: Optional tokenizer for exact token counting
            rules: Optional rules for recursive boundary finding
            method: Either 'static' or 'recursive'
            mode: Either 'suffix' or 'prefix'
            merge_context: Whether to merge context into chunk text
            inplace: Whether to modify chunks in place
            approximate: Whether to use approximate token counting
        """
        super().__init__(context_size)
        
        # Validate method
        if method not in ["static", "recursive"]:
            raise ValueError("method must be either 'static' or 'recursive'")
            
        # Validate mode
        if mode not in ["suffix", "prefix"]:
            raise ValueError("mode must be either 'suffix' or 'prefix'")
            
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
            self._tokenizer_backend = self._get_tokenizer_backend()
            self.approximate = approximate
        else:
            self.approximate = True
            
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

    def _count_tokens(self, text: str) -> int:
        """Unified token counting method."""
        if not text:
            return 0
        if hasattr(self, "tokenizer") and not self.approximate:
            return len(self.tokenizer.encode(text))
        return len(text) // self._AVG_CHAR_PER_TOKEN

    def _validate_chunks(self, chunks: List[Chunk]) -> None:
        """Validate chunk inputs."""
        if not chunks:
            return
            
        if len(set(type(chunk) for chunk in chunks)) > 1:
            raise ValueError("All chunks must be of the same type")

    def _find_boundary_with_rules(self, text: str, rule: RecursiveRules, 
                                direction: str = "backward") -> Optional[Tuple[str, int]]:
        """Generic boundary finding logic that works in both directions."""
        if not rule.delimiters:
            return None
            
        for delimiter in rule.delimiters:
            parts = text.split(delimiter)
            if len(parts) <= 1:
                continue
                
            if direction == "backward":
                # Build context from the end
                current_text = ""
                accumulated_parts = []
                
                for part in reversed(parts):
                    if not part:
                        continue
                        
                    test_text = part + delimiter + current_text
                    token_count = self._count_tokens(test_text)
                    
                    if token_count >= self.min_tokens:
                        if accumulated_parts:
                            final_text = part + delimiter + delimiter.join(accumulated_parts)
                        else:
                            final_text = test_text
                            
                        try:
                            start_pos = text.rindex(final_text)
                            return final_text, start_pos
                        except ValueError:
                            continue
                            
                    accumulated_parts.insert(0, part)
                    current_text = test_text
            else:
                # Build context from start
                current_text = ""
                accumulated_parts = []
                
                for part in parts:
                    if not part:
                        continue
                        
                    test_text = current_text + delimiter + part if current_text else part
                    token_count = self._count_tokens(test_text)
                    
                    if token_count >= self.min_tokens:
                        if accumulated_parts:
                            final_text = delimiter.join(accumulated_parts) + delimiter + part
                        else:
                            final_text = test_text
                            
                        try:
                            start_pos = text.index(final_text)
                            return final_text, start_pos + len(final_text)
                        except ValueError:
                            continue
                            
                    accumulated_parts.append(part)
                    current_text = test_text
                    
        return None

    def _find_primary_boundary_context(self, text: str) -> Optional[Tuple[str, int]]:
        """Find primary boundary context working backward."""
        if not self.rules or len(self.rules) == 0:
            return None
            
        return self._find_boundary_with_rules(text, self.rules[0], "backward")

    def _find_forward_boundary_context(self, text: str) -> Optional[Tuple[str, int]]:
        """Find primary boundary context working forward."""
        if not self.rules or len(self.rules) == 0:
            return None
            
        return self._find_boundary_with_rules(text, self.rules[0], "forward")

    def _get_hierarchical_context(self, chunk: Chunk) -> Optional[Context]:
        """Get context using hierarchical rules.
        
        First tries to find appropriate boundaries using primary rule,
        then falls back to secondary rules if needed.
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
    
    def _get_forward_hierarchical_context(self, chunk: Chunk) -> Optional[Context]:
        """Get context using hierarchical rules, working forwards."""
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

    def _prefix_overlap_token_exact(self, chunk: Chunk) -> Optional[Context]:
        """Calculate precise token-based overlap context using tokenizer."""
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

    def _prefix_overlap_token_approximate(self, chunk: Chunk) -> Optional[Context]:
        """Calculate approximate token-based overlap context."""
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

    def _get_prefix_overlap_context(self, chunk: Chunk) -> Optional[Context]:
        """Get appropriate overlap context based on chunk type."""
        if isinstance(chunk, SemanticChunk) or isinstance(chunk, SentenceChunk):
            return self._prefix_overlap_sentence(chunk)
        elif isinstance(chunk, Chunk):
            return self._prefix_overlap_token(chunk)
        else:
            raise ValueError(f"Unsupported chunk type: {type(chunk)}")

    def _prefix_overlap_token(self, chunk: Chunk) -> Optional[Context]:
        """Choose between exact or approximate token overlap calculation."""
        if self.approximate:
            return self._prefix_overlap_token_approximate(chunk)
        return self._prefix_overlap_token_exact(chunk)

    def _prefix_overlap_sentence(self, chunk: SentenceChunk) -> Optional[Context]:
            """Calculate overlap context based on sentences."""
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
        
    def _suffix_overlap_token_exact(self, chunk: Chunk) -> Optional[Context]:
        """Calculate precise token-based overlap context using tokenizer."""
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

    def _suffix_overlap_token_approximate(self, chunk: Chunk) -> Optional[Context]:
        """Calculate approximate token-based overlap context."""
        # Calculate desired context size
        context_tokens = min(self.context_size, chunk.token_count)

        # Estimate text length based on token ratio
        context_ratio = context_tokens / chunk.token_count
        char_length = int(len(chunk.text) * context_ratio)

        # Extract context text from start
        context_text = chunk.text[:char_length]

        return Context(
            text=context_text,
            token_count=context_tokens,
            start_index=chunk.start_index,
            end_index=chunk.start_index + char_length,
        )

    def _suffix_overlap_token(self, chunk: Chunk) -> Optional[Context]:
        """Choose between exact or approximate token overlap calculation."""
        if self.approximate:
            return self._suffix_overlap_token_approximate(chunk)
        return self._suffix_overlap_token_exact(chunk)

    def _get_suffix_overlap_context(self, chunk: Chunk) -> Optional[Context]:
        """Get appropriate overlap context based on chunk type."""
        if isinstance(chunk, SemanticChunk) or isinstance(chunk, SentenceChunk):
            return self._suffix_overlap_sentence(chunk)
        elif isinstance(chunk, Chunk):
            return self._suffix_overlap_token(chunk)
        else:
            raise ValueError(f"Unsupported chunk type: {type(chunk)}")

    def _suffix_overlap_sentence(self, chunk: SentenceChunk) -> Optional[Context]:
        """Calculate overlap context based on sentences from the start."""
        if not chunk.sentences:
            return None

        context_sentences = []
        total_tokens = 0

        # Add sentences from the start until we hit context_size
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

    def _refine_prefix_static(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks using static prefix overlap."""
        if not chunks:
            return chunks

        # Create new chunks if not modifying in place
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

    def _refine_suffix_static(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks using static suffix overlap."""
        if not chunks:
            return chunks

        # Create new chunks if not modifying in place
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
                refined_chunks[i].token_count = self._count_tokens(refined_chunks[i].text)

        return refined_chunks

    def _refine_prefix_hierarchical(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks using hierarchical prefix overlap."""
        if not chunks:
            return chunks

        # Create new chunks if not modifying in place
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
                refined_chunks[i].token_count = self._count_tokens(refined_chunks[i].text)

        return refined_chunks

    def _refine_suffix_hierarchical(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks using hierarchical suffix overlap."""
        if not chunks:
            return chunks

        # Create new chunks if not modifying in place
        if not self.inplace:
            refined_chunks = [chunk.copy() for chunk in chunks]
        else:
            refined_chunks = chunks

        # Process chunks before last
        for i in range(len(refined_chunks) - 1):
            # Get hierarchical context from next chunk
            context = self._get_forward_hierarchical_context(chunks[i + 1])
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
        """Refine chunks by adding appropriate context."""
        # Validate inputs first
        if not chunks:
            return chunks
        self._validate_chunks(chunks)
        
        # Choose appropriate refinement method
        if self.method == "recursive":
            if self.mode == "prefix":
                return self._refine_prefix_hierarchical(chunks)
            elif self.mode == "suffix":
                return self._refine_suffix_hierarchical(chunks)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
        else:
            # Original static overlap logic
            if self.mode == "prefix":
                return self._refine_prefix_static(chunks)
            elif self.mode == "suffix":
                return self._refine_suffix_static(chunks)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

    @classmethod
    def is_available(cls) -> bool:
        """Check if the OverlapRefinery is available."""
        return True