"""Recursive chunker."""
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from typing import Any, List, Optional, Union

from chonkie.chunker.base import BaseChunker
from chonkie.types import Chunk, RecursiveChunk, RecursiveRules, RecursiveLevel



class RecursiveChunker(BaseChunker):
    """Chunker that uses recursive rules to chunk text.
    
    Attributes:
        rules: The rules to use for chunking.
        chunk_size: The size of the chunks to return.
        
    """

    def __init__(self,
                 tokenizer: Union[str, Any] = "gpt2",
                 chunk_size: int = 512,
                 rules: RecursiveRules = RecursiveRules(), 
                 min_characters_per_chunk: int = 12
                 ) -> None:
        """Initialize the recursive chunker.

        Args:
            tokenizer: The tokenizer to use for encoding/decoding.
            chunk_size: The size of the chunks to return.
            rules: The rules to use for chunking.
            min_characters_per_chunk: The minimum number of characters per chunk.
            
        """
        super().__init__(tokenizer)
        self.rules = rules
        self.chunk_size = chunk_size
        self.min_characters_per_chunk = min_characters_per_chunk

    def _split_text(self,
                    text: str,
                    rule: RecursiveLevel, 
                    sep: str = "🦛") -> List[str]:
        """Split the text into chunks using the delimiters."""
        # At every delimiter, replace it with the sep   
        if rule.delimiters:
            for delimiter in rule.delimiters:
                text = text.replace(delimiter, delimiter + sep)

            # Split the text at the sep
            splits = [s for s in text.split(sep) if s != ""]

            # Usually a good idea to check if there are any splits that are too short in characters
            # and then merge them
            merged_splits = []
            for split in splits:
                if len(split) < self.min_characters_per_chunk:
                    merged_splits[-1] += split
                else:
                    merged_splits.append(split)
            splits = merged_splits

        elif rule.whitespace:
            splits = self._split_at_whitespace(text)
        else:
            splits = self._split_at_tokens(text)

        #NOTE: Usually some splits will be very short and not meaningful, but 
        # we can at this point assume that the merge will fix this later on. 
        return splits
    
    def _split_at_whitespace(self,
                             text: str) -> List[str]:
        """Split the text at whitespace."""
        return text.split(' ')
    
    def _split_at_tokens(self,
                         text: str) -> List[str]:
        """Split the text at tokens."""
        tokens = self._encode(text)

        # Split the tokens at the chunk size
        token_splits = [tokens[i:i+self.chunk_size] for i in range(0, len(tokens), self.chunk_size)]
        
        # Decode the tokens back to text
        splits = self._decode_batch(token_splits)
        return splits

    def _merge_splits(self,
                       splits: List[str],
                       token_counts: List[int],
                       combine_with_whitespace: bool = False) -> List[str]:
        """Merge splits that are too short."""
        # If there are no splits or token counts, return an empty list
        if not splits or not token_counts:
            return [], []
        
        # If the number of splits and token counts does not match, raise an error
        if len(splits) != len(token_counts):
            raise ValueError("The number of splits and token counts does not match.")

        # Usually the splits can be smaller than the chunk size; if not, 
        # we can just return the splits
        if all(tc > self.chunk_size for tc in token_counts):
            return splits, token_counts
        
        # If the splits are too short, merge them
        merged = []

        if not combine_with_whitespace:
            cumulative_token_counts = list(accumulate([0] + token_counts, lambda x, y: x + y))
        else:   
            cumulative_token_counts = list(accumulate([0] + token_counts, lambda x, y: x + y + 1)) # Add 1 for the whitespace

        current_index = 0
        merged_token_counts = []

        # Use bisect_left to find the index to merge at 
        while current_index < len(splits):
            current_token_count = cumulative_token_counts[current_index]
            required_token_count = current_token_count + self.chunk_size
            # print(current_index, current_token_count, required_token_count)
            
            # Find the index to merge at
            index = min(bisect_left(cumulative_token_counts, required_token_count, lo=current_index) - 1, len(splits))
            # print(f"index: {index}\n")
            
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
            # print(f"merged_token_counts: {merged_token_counts}\n")
            
            # Update the current index
            current_index = index

        return merged, merged_token_counts

    @lru_cache(maxsize=4096)
    def _get_token_count(self,
                         text: str) -> int:
        """Get the token count of the text."""
        CHARS_PER_TOKEN = 6.5  # Avg. char per token for llama3 is b/w 6-7
        estimate = max(1, len(text) // CHARS_PER_TOKEN)
        if estimate > self.chunk_size:
            return self.chunk_size + 1
        else:
            return self._count_tokens(text)

    def _create_chunk(self,
                      text: str, 
                      token_count: int, 
                      level: int,
                      full_text: Optional[str] = None) -> RecursiveChunk:
        """Create a chunk."""
        if full_text is None:
            full_text = text
        try:
            start_index = full_text.index(text)
            end_index = start_index + len(text)
        except Exception as e:
            print(f"Error getting start_index and end_index: {e}"
                  f"text: {text}\n"
                  f"full_text: {full_text}\n")
            start_index = 0
            end_index = 0
            
        return RecursiveChunk(text=text,
                              start_index=start_index,
                              end_index=end_index,
                              token_count=token_count,
                              level=level)

    def _recursive_chunk(self, 
                        text: str,
                        level: int = 0, 
                        full_text: Optional[str] = None) -> List[RecursiveChunk]:
        """Recursive chunking logic."""
        # First make sure the text is not empty
        if not text:
            return []
        
        # If level is out of bounds, return the text as a chunk
        if level >= len(self.rules):
            return [self._create_chunk(text, self._get_token_count(text), level, full_text)]

        # If full_text is not provided, use the text
        if full_text is None:
            full_text = text
        
        # Get the current level
        rule = self.rules[level]

        # Split the text at the current level
        splits = self._split_text(text, rule)

        # Get the token counts for the splits
        token_counts = [self._get_token_count(split) for split in splits]

        # Merge the splits
        if rule.delimiters is None and not rule.whitespace:
            # If the level is at tokens, we don't need to merge the splits
            merged, merged_token_counts = splits, token_counts

        elif rule.delimiters is None and rule.whitespace:
            # If the level is at tokens and whitespace, we need to merge the splits
            merged, merged_token_counts = self._merge_splits(splits, token_counts, combine_with_whitespace=True)
        else:
            # If the level is at sentences, we need to merge the splits
            merged, merged_token_counts = self._merge_splits(splits, token_counts, combine_with_whitespace=False)

        # Recursively chunk the merged splits if they are too long
        chunks = []
        for split, token_count in zip(merged, merged_token_counts):
            if token_count > self.chunk_size:
                chunks.extend(self._recursive_chunk(split, level + 1, full_text))
            else:
                if rule.delimiters is None and not rule.whitespace:
                    # NOTE: This is a hack to get the decoded text, since merged = splits = token_splits
                    # And we don't want to encode/decode the text again, that would be inefficient
                    decoded_text = "".join(merged)
                    chunks.append(self._create_chunk(split, token_count, level, decoded_text))
                else:
                    chunks.append(self._create_chunk(split, token_count, level, full_text))

        return chunks


    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the text."""
        return self._recursive_chunk(text, level=0, full_text=text)


    def __repr__(self) -> str:
        """Get a string representation of the recursive chunker."""
        return (f"RecursiveChunker(rules={self.rules}, "
                f"chunk_size={self.chunk_size}, "
                f"min_characters_per_chunk={self.min_characters_per_chunk})")
    
    def __str__(self) -> str:
        """Get a string representation of the recursive chunker."""
        return (f"RecursiveChunker(rules={self.rules}, "
                f"chunk_size={self.chunk_size}, "
                f"min_characters_per_chunk={self.min_characters_per_chunk})")
