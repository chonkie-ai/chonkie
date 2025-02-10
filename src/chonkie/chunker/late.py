"""Class definition for Late Chunking."""

import importlib
from bisect import bisect_left
from itertools import accumulate
from typing import TYPE_CHECKING, List, Literal, Union

from chonkie.embeddings import BaseEmbeddings, SentenceTransformerEmbeddings
from chonkie.types import LateChunk, Sentence

from .base import BaseChunker

if TYPE_CHECKING:
    import numpy as np

class LateChunker(BaseChunker):
    """Class for Late Chunking.

    In late chunking, we first take the embeddings of the entire text,
    after which we split the text into chunks. When we split the text, 
    we can use the embeddings generated earlier, mean pool them and 
    use them as the sentence embeddings for the chunks. 

    This class particularly implements the Sentence-style LateChunking 
    approach by changing the way we prepare the sentences before the 
    chunking. 

    Args:
        embedding_model: The embedding model to use for the LateChunker
        mode: Mode of the LateChunker to split the text in
        chunk_size: Maximum number of tokens per chunk
        min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
        min_characters_per_sentence: Minimum number of characters per sentence (defaults to 12)
        delim: Delimiters to split sentences on
        include_delim: Whether to include the delimiters in the sentences
        **kwargs: Additional keyword arguments to pass to the EmbeddingModel
        
    """  

    def __init__(self,
                 embedding_model: Union[str, BaseEmbeddings] = "all-minilm-l6-v2", 
                 mode: str = "sentence",
                 chunk_size: int = 512, 
                 min_sentences_per_chunk: int = 1, 
                 min_characters_per_sentence: int = 12,
                 approximate: bool = True,
                 delim: Union[str, List[str]] = ['.', '!', '?', '\n'],
                 include_delim: Union[Literal["prev", "next"], None] = "prev",
                 **kwargs
                 ) -> None:
        """Initialise the LateChunker."""
        # Assign the values if they make sense
        if mode not in ['token', 'sentence']:
            raise ValueError(
                "Mode must be one of the following: ['token', 'sentence']"
            )
        if chunk_size <= 0: 
            raise ValueError("chunk_size must be a positive non-zero value!")
        if min_sentences_per_chunk < 0 and mode == 'sentence':
            raise ValueError(f"min_sentences_per_chunk was assigned {min_sentences_per_chunk}; but it must be non-negative value!")
        if min_characters_per_sentence <= 0:
            raise ValueError("min_characters_per_sentence must be a positive non-zero value!")
        if type(delim) not in [str, list]:
            raise TypeError("delim must be of type str or list of str")
        
        self.mode = mode
        self.chunk_size = chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.approximate = approximate
        self.delim = delim
        self.include_delim = include_delim
        self.sep = '🦛'

        # Initialise the embeddings via AutoEmbeddings
        if isinstance(embedding_model, BaseEmbeddings): 
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            from chonkie.embeddings.auto import AutoEmbeddings
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model, **kwargs)
        else:
            raise ValueError(
                "ooops! seems like your embedding model is of the wrong type " +
                "currently, this class only supports BaseEmbeddings objects or str objects "+
                f"and it received {type(embedding_model)} object!"
            )

        # Check if the embedding model has been assigned properly
        if self.embedding_model is None:
            raise ImportError(
                "embedding_model is not a valid embedding model",
                "Please install the `st` extra to use this feature"
            )
        
        if type(self.embedding_model) != SentenceTransformerEmbeddings:
            raise ValueError("LateChunker (currently) only works with SentenceTransformerEmbeddings", 
                             "Please install the `st` extra to use this feature")
        
        # Import numpy here as to not import it when it's not needed
        if importlib.util.find_spec('numpy') is not None:
            global np
            import numpy as np

        # Keeping the tokenizer the same as the sentence model is important 
        # for the semantic meaning to be calculated properly
        super().__init__(self.embedding_model.get_tokenizer_or_token_counter())

        # Remove the multiprocessing flag from the base class
        self._use_multiprocessing = False

    def _create_token_chunks(self,
                            chunk_texts: List[str],
                            token_counts: List[int],
                            decoded_text: str,
                            ) -> List[LateChunk]:
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
                LateChunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count,
                )
            )
            current_index = end_index
        return chunks

    def _token_chunk(self, text: str) -> List[LateChunk]:
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
                0, len(text_tokens), self.chunk_size
            )
        ]
        token_counts = [
            len(toks) for toks in token_groups
        ]  # get the token counts; it's prolly chunk_size, but len doesn't take too long

        chunk_texts = self._decode_batch(
            token_groups
        )  # decrease the time by decoding in one go (?)

        chunks = self._create_token_chunks(chunk_texts, token_counts, decoded_text)

        return chunks


    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        """Fast sentence splitting while maintaining accuracy.

        This method is faster than using regex for sentence splitting and is more accurate than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences
            delim: Delimiters to split sentences on
            sep: Separator to use when splitting sentences

        Returns:
            List of sentences

        """
        t = text
        for c in self.delim:
            if self.include_delim == "prev":
                t = t.replace(c, c + self.sep)
            elif self.include_delim == "next":
                t = t.replace(c, self.sep + c)
            else:
                t = t.replace(c, self.sep)

        # Initial split
        splits = [s for s in t.split(self.sep) if s != ""]
        # print(splits)

        # Combine short splits with previous sentence
        current = ""
        sentences = []
        for split in splits:
            if len(split) < self.min_characters_per_sentence:
                current += split
            elif current:
                current += split
                sentences.append(current)
                current = ""
            else:
                sentences.append(split)
            
            # At any point if the current sentence is longer than the min_characters_per_sentence,
            # add it to the sentences
            if len(current) >= self.min_characters_per_sentence:
                sentences.append(current)
                current = ""

        # If there is a current split, add it to the sentences
        if current:
            sentences.append(current)

        return sentences

    def _get_token_counts(self, sentences: List[str]) -> List[int]:
        """Get token counts for a list of sentences in batch.

        Args:
            sentences: List of sentences

        Returns:
            List of token counts for each sentence

        """
        # Batch encode all sentences at once
        encoded_sentences = self._encode_batch(sentences)
        return [len(encoded) for encoded in encoded_sentences]

    def _estimate_token_counts(self, text: str) -> int:
        """Estimate token count using character length."""
        CHARS_PER_TOKEN = 6.0  # Avg. char per token for llama3 is b/w 6-7
        if type(text) is str:
            return max(1, int(len(text) / CHARS_PER_TOKEN))
        elif type(text) is list and type(text[0]) is str:
            return [max(1, int(len(t) / CHARS_PER_TOKEN)) for t in text]
        else:
            raise ValueError(
                f"Unknown type passed to _estimate_token_count: {type(text)}"
            )
    def _get_feedback(self, estimate: int, actual: int) -> float:
        """Validate against the actual token counts and correct the estimates."""
        feedback = 1 - ((estimate - actual) / estimate)
        return feedback
    
    def _prepare_sentences(self, text: str) -> List[Sentence]:
        """Split text into sentences and calculate token counts for each sentence.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of Sentence objects

        """
        # Split text into sentences
        sentence_texts = self._split_sentences(text)
        if not sentence_texts:
            return []

        # Calculate positions once
        positions = []
        current_pos = 0
        for sent in sentence_texts:
            positions.append(current_pos)
            current_pos += len(sent)  # No +1 space because sentences are already separated by spaces

        if not self.approximate:
            # Get accurate token counts in batch
            token_counts = self._get_token_counts(sentence_texts)
        else:
            # Estimate token counts using character length
            token_counts = self._estimate_token_counts(sentence_texts)

        # Create sentence objects
        return [
            Sentence(
                text=sent, start_index=pos, end_index=pos + len(sent), token_count=count
            )
            for sent, pos, count in zip(sentence_texts, positions, token_counts)
        ]
    
    def _create_sentence_chunk(self, sentences: List[Sentence], token_count: int) -> LateChunk:
        """Create a chunk from a list of sentences.

        Args:
            sentences: List of sentences to create chunk from
            token_count: Total token count for the chunk

        Returns:
            Chunk object

        """
        chunk_text = "".join([sentence.text for sentence in sentences])
        return LateChunk(
            text=chunk_text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
            sentences=sentences,
        )

    def _sentence_chunk(self, text: str) -> List[LateChunk]:
        """Chunk the text into sentences."""
        """Split text into overlapping chunks based on sentences while respecting token limits.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        # Get prepared sentences with token counts
        sentences = self._prepare_sentences(text)  # 28mus
        if not sentences:
            return []

        # Pre-calculate cumulative token counts for bisect
        # Add 1 token for spaces between sentences
        token_sums = list(
            accumulate(
                [s.token_count for s in sentences], lambda a, b: a + b, initial=0
            )
        )

        chunks = []
        feedback = 1.0
        pos = 0

        while pos < len(sentences):
            # use updated feedback on the token sums
            token_sums = [int(s * feedback) for s in token_sums]

            # Use bisect_left to find initial split point
            target_tokens = token_sums[pos] + self.chunk_size
            split_idx = bisect_left(token_sums, target_tokens) - 1
            split_idx = min(split_idx, len(sentences))

            # Ensure we include at least one sentence beyond pos
            split_idx = max(split_idx, pos + 1)

            # Handle minimum sentences requirement
            if split_idx - pos < self.min_sentences_per_chunk:
                split_idx = pos + self.min_sentences_per_chunk

            # Get the estimated token count
            estimate = token_sums[split_idx] - token_sums[pos]

            # Get candidate sentences and verify actual token count
            chunk_sentences = sentences[pos:split_idx]
            chunk_text = "".join(s.text for s in chunk_sentences)
            actual = len(self._encode(chunk_text))

            # Given the actual token_count and the estimate, get a feedback value for the next loop
            feedback = self._get_feedback(estimate, actual)
            # print(f"Estimate: {estimate} Actual: {actual} feedback: {feedback}")

            # Back off one sentence at a time if we exceeded chunk size
            while (
                actual > self.chunk_size
                and len(chunk_sentences) > self.min_sentences_per_chunk
            ):
                split_idx -= 1
                chunk_sentences = sentences[pos:split_idx]
                chunk_text = "".join(s.text for s in chunk_sentences)
                actual = len(self._encode(chunk_text))

            chunks.append(self._create_sentence_chunk(chunk_sentences, actual))
            pos = split_idx

        return chunks

    def _get_chunks(self, text: str) -> List[LateChunk]:
        """Get chunks from the text."""
        if self.mode == 'token':
            return self._token_chunk(text)
        elif self.mode == 'sentence':
            return self._sentence_chunk(text)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _embedding_split(self, token_embeddings: "np.ndarray", token_counts: List[int]) -> List["np.ndarray"]:
        """Split the embedding into chunks."""
        embedding_splits = []
        current_index = 0
        for i, token_count in enumerate(token_counts):
            if i != len(token_counts) - 1:
                embedding_splits.append(token_embeddings[current_index:current_index+token_count])
                current_index += token_count
            else:
                embedding_splits.append(token_embeddings[current_index:])
        return embedding_splits
        
    def _mean_pool(self, embeddings: "np.ndarray") -> "np.ndarray":
        """Mean pool the embeddings."""
        # Assuming that numpy is installed and imported as np
        # which is the case for the SentenceTransformerEmbeddings
        return np.mean(embeddings, axis=0)

    def chunk(self, text: str) -> List[LateChunk]:
        """Chunk the text via Late Chunking."""
        # Get the chunks first
        chunks = self._get_chunks(text)
        token_counts = [chunk.token_count for chunk in chunks]

        # NOTE: A known issue with the SentenceTransformerEmbeddings is that it doesn't
        # allow getting the token embeddings without adding special tokens. So the token
        # embeddings are not exactly the same as the token embeddings of the text
        # Additionally, token counts are not exactly the same as the token counts of the text
        # because the tokenizer encodes the text differently if the text is split into chunks


        # Get the token embeddings for the entire text
        token_embeddings = self.embedding_model.embed_as_tokens(text)  # Shape: (n_tokens, embedding_dim)
        chunk_token_embeddings = self._embedding_split(token_embeddings, token_counts)  # Shape: (n_chunks, n_tokens, embedding_dim)

        # Get the chunk embeddings by averaging the token embeddings
        chunk_embeddings = [self._mean_pool(token_emb) for token_emb in chunk_token_embeddings] # Shape: (n_chunks, embedding_dim)

        # Add the chunk embeddings to the chunks
        for chunk, embedding in zip(chunks, chunk_embeddings):
            chunk.embedding = embedding

        return chunks
