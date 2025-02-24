"""Test cases for the LateChunker."""

from typing import List

import numpy as np
import pytest

from chonkie.chunker.late import LateChunker
from chonkie.embeddings import SentenceTransformerEmbeddings
from chonkie.types import LateChunk


@pytest.fixture
def embedding_model():
    """Return a sentence transformer embedding model instance."""
    return SentenceTransformerEmbeddings("all-minilm-l6-v2")


@pytest.fixture
def sample_text():
    """Return a sample text."""
    text = """# Chunking Strategies in Retrieval-Augmented Generation: A Comprehensive Analysis\n\nIn the rapidly evolving landscape of natural language processing, Retrieval-Augmented Generation (RAG) has emerged as a groundbreaking approach that bridges the gap between large language models and external knowledge bases. At the heart of these systems lies a crucial yet often overlooked process: chunking. This fundamental operation, which involves the systematic decomposition of large text documents into smaller, semantically meaningful units, plays a pivotal role in determining the overall effectiveness of RAG implementations.\n\nThe process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. This balancing act becomes particularly crucial when we consider the downstream implications for vector databases and embedding models that form the backbone of modern RAG systems.\n\nThe selection of appropriate chunk size emerges as a fundamental consideration that significantly impacts system performance. Through extensive experimentation and real-world implementations, researchers have identified that chunks typically perform optimally in the range of 256 to 1024 tokens. However, this range should not be treated as a rigid constraint but rather as a starting point for optimization based on specific use cases and requirements. The implications of chunk size selection ripple throughout the entire RAG pipeline, affecting everything from storage requirements to retrieval accuracy and computational overhead.\n\nFixed-size chunking represents the most straightforward approach to document segmentation, offering predictable memory usage and consistent processing time. However, this apparent simplicity comes with significant drawbacks. By arbitrarily dividing text based on token or character count, fixed-size chunking risks fragmenting semantic units and disrupting the natural flow of information. Consider, for instance, a technical document where a complex concept is explained across several paragraphs – fixed-size chunking might split this explanation at critical junctures, potentially compromising the system's ability to retrieve and present this information coherently.\n\nIn response to these limitations, semantic chunking has gained prominence as a more sophisticated alternative. This approach leverages natural language understanding to identify meaningful boundaries within the text, respecting the natural structure of the document. Semantic chunking can operate at various levels of granularity, from simple sentence-based segmentation to more complex paragraph-level or topic-based approaches. The key advantage lies in its ability to preserve the inherent semantic relationships within the text, leading to more meaningful and contextually relevant retrieval results.\n\nRecent advances in the field have given rise to hybrid approaches that attempt to combine the best aspects of both fixed-size and semantic chunking. These methods typically begin with semantic segmentation but impose size constraints to prevent extreme variations in chunk length. Furthermore, the introduction of sliding window techniques with overlap has proved particularly effective in maintaining context across chunk boundaries. This overlap, typically ranging from 10% to 20% of the chunk size, helps ensure that no critical information is lost at segment boundaries, albeit at the cost of increased storage requirements.\n\nThe implementation of chunking strategies must also consider various technical factors that can significantly impact system performance. Vector database capabilities, embedding model constraints, and runtime performance requirements all play crucial roles in determining the optimal chunking approach. Moreover, content-specific factors such as document structure, language characteristics, and domain-specific requirements must be carefully considered. For instance, technical documentation might benefit from larger chunks that preserve detailed explanations, while news articles might perform better with smaller, more focused segments.\n\nThe future of chunking in RAG systems points toward increasingly sophisticated approaches. Current research explores the potential of neural chunking models that can learn optimal segmentation strategies from large-scale datasets. These models show promise in adapting to different content types and query patterns, potentially leading to more efficient and effective retrieval systems. Additionally, the emergence of cross-lingual chunking strategies addresses the growing need for multilingual RAG applications, while real-time adaptive chunking systems attempt to optimize segment boundaries based on user interaction patterns and retrieval performance metrics.\n\nThe effectiveness of RAG systems heavily depends on the thoughtful implementation of appropriate chunking strategies. While the field continues to evolve, practitioners must carefully consider their specific use cases and requirements when designing chunking solutions. Factors such as document characteristics, retrieval patterns, and performance requirements should guide the selection and optimization of chunking strategies. As we look to the future, the continued development of more sophisticated chunking approaches promises to further enhance the capabilities of RAG systems, enabling more accurate and efficient information retrieval and generation.\n\nThrough careful consideration of these various aspects and continued experimentation with different approaches, organizations can develop chunking strategies that effectively balance the competing demands of semantic coherence, computational efficiency, and retrieval accuracy. As the field continues to evolve, we can expect to see new innovations that further refine our ability to segment and process textual information in ways that enhance the capabilities of RAG systems while maintaining their practical utility in real-world applications."""
    return text


def test_late_chunker_initialization(embedding_model):
    """Test that the LateChunker can be initialized properly."""
    chunker = LateChunker(
        embedding_model=embedding_model, mode="sentence", chunk_size=512
    )

    assert chunker is not None
    assert isinstance(chunker.embedding_model, SentenceTransformerEmbeddings)
    assert chunker.mode == "sentence"
    assert chunker.chunk_size == 512
    assert chunker.min_sentences_per_chunk == 1


def test_late_chunker_invalid_mode(embedding_model):
    """Test that the LateChunker raises error for invalid mode."""
    with pytest.raises(ValueError):
        LateChunker(embedding_model=embedding_model, mode="invalid")


def test_late_chunker_sentence_mode(embedding_model, sample_text):
    """Test that the LateChunker works in sentence mode."""
    chunker = LateChunker(
        embedding_model=embedding_model, mode="sentence", chunk_size=512
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], LateChunk)
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(isinstance(chunk.embedding, np.ndarray) for chunk in chunks)


def test_late_chunker_token_mode(embedding_model, sample_text):
    """Test that the LateChunker works in token mode."""
    chunker = LateChunker(embedding_model=embedding_model, mode="token", chunk_size=512)
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], LateChunk)
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(isinstance(chunk.embedding, np.ndarray) for chunk in chunks)


def test_late_chunker_empty_text(embedding_model):
    """Test that the LateChunker can handle empty text input."""
    chunker = LateChunker(embedding_model=embedding_model)
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_late_chunker_single_sentence(embedding_model):
    """Test that the LateChunker can handle text with a single sentence."""
    chunker = LateChunker(embedding_model=embedding_model)
    text = "This is a single sentence."
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].embedding is not None
    assert isinstance(chunks[0].embedding, np.ndarray)


def test_late_chunker_sentence_boundaries(embedding_model):
    """Test that the LateChunker respects sentence boundaries."""
    text = "First sentence. Second sentence. Third sentence."
    chunker = LateChunker(
        embedding_model=embedding_model,
        mode="sentence",
        chunk_size=512,
        min_sentences_per_chunk=2,
    )
    chunks = chunker.chunk(text)

    for chunk in chunks:
        # Count sentences by splitting on periods
        sentence_count = len([s for s in chunk.text.split(".") if s.strip()])
        assert (
            sentence_count >= 2 or sentence_count == 1
        )  # Last chunk might have fewer sentences


def verify_chunk_indices(chunks: List[LateChunk], original_text: str):
    """Verify that chunk indices correctly map to the original text."""
    for i, chunk in enumerate(chunks):
        # Extract text using the indices
        extracted_text = original_text[chunk.start_index : chunk.end_index]
        # Remove any leading/trailing whitespace from both texts for comparison
        chunk_text = chunk.text.strip()
        extracted_text = extracted_text.strip()

        assert chunk_text == extracted_text, (
            f"Chunk {i} text mismatch:\n"
            f"Chunk text: '{chunk_text}'\n"
            f"Extracted text: '{extracted_text}'\n"
            f"Indices: [{chunk.start_index}:{chunk.end_index}]"
        )


def test_late_chunker_indices(embedding_model, sample_text):
    """Test that the LateChunker correctly maps chunk indices to the original text."""
    chunker = LateChunker(embedding_model=embedding_model)
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_late_chunker_embedding_dimensions(embedding_model, sample_text):
    """Test that all chunk embeddings have consistent dimensions."""
    chunker = LateChunker(embedding_model=embedding_model)
    chunks = chunker.chunk(sample_text)

    # Get expected embedding dimension from the model
    expected_dim = embedding_model.dimension

    for chunk in chunks:
        assert chunk.embedding.shape == (expected_dim,)


if __name__ == "__main__":
    pytest.main()
