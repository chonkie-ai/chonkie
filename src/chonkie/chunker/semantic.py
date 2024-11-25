from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from .base import BaseChunker
from .sentence import Sentence, SentenceChunk

from chonkie.embeddings.base import BaseEmbeddings

@dataclass
class SemanticSentence(Sentence):
    """Dataclass representing a semantic sentence with metadata.

    All attributes are read-only via slots for performance reasons.

    Attributes:
        text: The text content of the sentence
        start_index: The starting index of the sentence in the original text
        end_index: The ending index of the sentence in the original text
        token_count: The number of tokens in the sentence
        embedding: The sentence embedding
    """
    embedding: Optional[np.ndarray]
    
    # Only define new slots, not the ones inherited from Sentence
    __slots__ = ['embedding',]
    
    def __init__(self, text: str, start_index: int, end_index: int, 
                 token_count: int, embedding: Optional[np.ndarray] = None):
        super().__init__(text, start_index, end_index, token_count)
        object.__setattr__(self, 'embedding', embedding if embedding is not None else None)

@dataclass
class SemanticChunk(SentenceChunk):
    """SemanticChunk dataclass representing a semantic chunk with metadata.

    All attributes are read-only via slots for performance reasons.

    Attributes:
        text: The text content of the chunk
        start_index: The starting index of the chunk in the original text
        end_index: The ending index of the chunk in the original text
        token_count: The number of tokens in the chunk
        sentences: List of SemanticSentence objects in the chunk
    """
    sentences: List[SemanticSentence] = field(default_factory=list)
    
    # No new slots needed since we're just overriding the sentences field
    __slots__ = []
    
    def __init__(self, text: str, start_index: int, end_index: int, 
                 token_count: int, sentences: List[SemanticSentence] = None):
        super().__init__(text, start_index, end_index, token_count)
        object.__setattr__(self, 'sentences', sentences if sentences is not None else [])

class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-8M",
        similarity_threshold: Optional[float] = None,
        similarity_percentile: Optional[float] = None,
        chunk_size: int = 512,
        initial_sentences: int = 1, 
        min_chunk_size: int = 2
    ):
        """Initialize the SemanticChunker.

        SemanticChunkers split text into semantically coherent chunks using embeddings.

        Args:
            embedding_model: Name of the sentence-transformers model to load
            chunk_size: Maximum tokens allowed per chunk
            similarity_threshold: Absolute threshold for semantic similarity (0-1)
            similarity_percentile: Percentile threshold for similarity (0-100)
            initial_sentences: Number of sentences to start each chunk with

        Raises:
            ValueError: If parameters are invalid
            ImportError: If required dependencies aren't installed
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if similarity_threshold is not None and (
            similarity_threshold < 0 or similarity_threshold > 1
        ):
            raise ValueError("similarity_threshold must be between 0 and 1")
        if similarity_percentile is not None and (
            similarity_percentile < 0 or similarity_percentile > 100
        ):
            raise ValueError("similarity_percentile must be between 0 and 100")
        if similarity_threshold is not None and similarity_percentile is not None:
            raise ValueError(
                "Cannot specify both similarity_threshold and similarity_percentile"
            )
        if similarity_threshold is None and similarity_percentile is None:
            similarity_percentile = 80
            raise Warning(
                "No similarity threshold specified. Defaulting to 80th percentile."
            ) #TODO: Change this to be a non-blocking warning
        if min_chunk_size < 1:
            raise ValueError("min_chunk_size must be at least 1")

        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.similarity_percentile = similarity_percentile
        self.initial_sentences = initial_sentences
        self.min_chunk_size = min_chunk_size

        if isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            from chonkie.embeddings.auto import AutoEmbeddings
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            raise ValueError("embedding_model must be a string or BaseEmbeddings instance")        
        
        # Keeping the tokenizer the same as the sentence model is important
        # for the group semantic meaning to be calculated properly
        super().__init__(self.embedding_model.get_tokenizer_or_token_counter())
    
    def _split_sentences(self,
                        text: str,
                        delim: Union[str, List[str]]=['.', '!', '?', '\n'],
                        sep: str="🦛") -> List[str]:
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
        for c in delim:
            t = t.replace(c, c + sep)
        
        # Initial split
        splits = [s for s in t.split(sep) if s != '']
        # print(splits)
        
        # Combine short splits with previous sentence
        sentences = []
        current = ""
        
        for s in splits:
            if len(s.strip()) < (self.min_chunk_size * 5):
                current += s
            else:
                if current:
                    sentences.append(current)
                current = s
        
        if current:
            sentences.append(current)
            
        return sentences

    def _compute_similarity_threshold(self, all_similarities: List[float]) -> float:
        """Compute similarity threshold based on percentile if specified."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        else:
            return float(np.percentile(all_similarities, self.similarity_percentile))

    def _prepare_sentences(self, text: str) -> List[Sentence]:
        """Prepare sentences with precomputed information.

        Args:
            text: Input text to be processed

        Returns:
            List of Sentence objects with precomputed token counts and embeddings
        """
        if not text.strip():
            return []

        # Split text into sentences
        raw_sentences = self._split_sentences(text)

        # Get start and end indices for each sentence
        sentence_indices = []
        current_idx = 0
        for sent in raw_sentences:
            start_idx = text.find(sent, current_idx)
            end_idx = start_idx + len(sent)
            sentence_indices.append((start_idx, end_idx))
            current_idx = end_idx

        # Batch compute embeddings for all sentences
        embeddings = self.embedding_model.embed_batch(raw_sentences)

        # Batch compute token counts
        token_counts = self._count_tokens_batch(raw_sentences)
        
        # Create Sentence objects with all precomputed information
        sentences = [
            SemanticSentence(
                text=sent,
                start_index=start_idx,
                end_index=end_idx,
                token_count=count,
                embedding=embedding,
            )
            for sent, (start_idx, end_idx), count, embedding in zip(
                raw_sentences, sentence_indices, token_counts, embeddings
            )
        ]

        # Get or compute similarity threshold
        if self.similarity_threshold is None:
            # Compute all pairwise similarities
            all_similarities = [
                self._get_semantic_similarity(
                    sentences[i].embedding, sentences[i + 1].embedding
                )
                for i in range(len(sentences) - 1)
            ]
            self.similarity_threshold = float(
                np.percentile(all_similarities, self.similarity_percentile)
            )
        return sentences

    def _get_semantic_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        similarity = self.embedding_model.similarity(embedding1, embedding2)
        return similarity

    def _compute_group_embedding(self, sentences: List[Sentence]) -> np.ndarray:
        """Compute mean embedding for a group of sentences."""
        return np.divide(
            np.sum([(sent.embedding * sent.token_count) for sent in sentences], axis=0),
            np.sum([sent.token_count for sent in sentences]),
            dtype=np.float32,
        )

    def _group_sentences(self, sentences: List[Sentence]) -> List[List[Sentence]]:
        """Group sentences based on semantic similarity, ignoring token count limits.

        Args:
            sentences: List of Sentence objects with precomputed embeddings

        Returns:
            List of sentence groups, where each group is semantically coherent
        """
        if len(sentences) <= self.initial_sentences:
            return [sentences]

        groups = []
        current_group = sentences[: self.initial_sentences]
        current_embedding = self._compute_group_embedding(current_group)

        for sentence in sentences[self.initial_sentences :]:
            # Compare new sentence against mean embedding of entire current group
            similarity = self._get_semantic_similarity(
                current_embedding, sentence.embedding
            )

            if similarity >= self.similarity_threshold:
                # Add to current group
                current_group.append(sentence)
                # Update mean embedding
                current_embedding = self._compute_group_embedding(current_group)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_embedding = sentence.embedding

        # Add final group
        if current_group:
            groups.append(current_group)

        return groups

    def _create_chunk(
        self, sentences: List[Sentence], similarity_scores: List[float] = None
    ) -> SemanticChunk:
        """Create a chunk from a list of sentences."""
        if not sentences:
            raise ValueError("Cannot create chunk from empty sentence list")

        # Compute chunk text and token count from sentences
        text = "".join(sent.text for sent in sentences)
        token_count = sum(sent.token_count for sent in sentences) + (
            len(sentences) - 1
        )  # Add spaces

        return SemanticChunk(
            text=text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
            sentences=sentences,
        )

    def _split_chunks(
        self, sentence_groups: List[List[Sentence]]
    ) -> List[SemanticChunk]:
        """Split sentence groups into chunks that respect chunk_size.

        Args:
            sentence_groups: List of semantically coherent sentence groups

        Returns:
            List of SemanticChunk objects
        """
        chunks = []

        for group in sentence_groups:
            current_chunk_sentences = []
            current_tokens = 0

            for sentence in group:
                test_tokens = (
                    current_tokens
                    + sentence.token_count
                    + (1 if current_chunk_sentences else 0)
                )

                if test_tokens <= self.chunk_size:
                    # Add to current chunk
                    current_chunk_sentences.append(sentence)
                    current_tokens = test_tokens
                else:
                    # Create chunk if we have sentences
                    if current_chunk_sentences:
                        chunks.append(self._create_chunk(current_chunk_sentences))

                    # Start new chunk with current sentence
                    current_chunk_sentences = [sentence]
                    current_tokens = sentence.token_count

            # Create final chunk for this group
            if current_chunk_sentences:
                chunks.append(self._create_chunk(current_chunk_sentences))

        return chunks

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into semantically coherent chunks using two-pass approach.

        First groups sentences by semantic similarity, then splits groups to respect
        chunk_size while maintaining sentence boundaries.

        Args:
            text: Input text to be chunked

        Returns:
            List of SemanticChunk objects containing the chunked text and metadata
        """
        if not text.strip():
            return []

        # Prepare sentences with precomputed information
        sentences = self._prepare_sentences(text)
        if len(sentences) < self.initial_sentences:
            return [self._create_chunk(sentences)]

        # First pass: Group sentences by semantic similarity
        sentence_groups = self._group_sentences(sentences)

        # Second pass: Split groups into size-appropriate chunks
        chunks = self._split_chunks(sentence_groups)

        return chunks

    def __repr__(self) -> str:
        threshold_info = (
            f"similarity_threshold={self.similarity_threshold}"
            if self.similarity_threshold is not None
            else f"similarity_percentile={self.similarity_percentile}"
        )
        return (
            f"SemanticChunker(chunk_size={self.chunk_size}, "
            f"{threshold_info}, "
            f"initial_sentences={self.initial_sentences})"
        )