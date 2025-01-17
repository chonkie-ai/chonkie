"""Tests for the SDPM (Semantic Density Peak Mapping) Chunker.

This module contains test cases for the SDPMChunker class, which implements
a chunking strategy based on semantic density and peak mapping. The tests verify:

- Basic chunking functionality with simple text
- Handling of complex markdown formatted text
- Proper semantic chunk generation
- Integration with embedding models
- Edge cases and boundary conditions

"""

import pytest

from chonkie.chunker.sdpm import SDPMChunker
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chonkie.types import SemanticChunk


@pytest.fixture
def sample_text():
    """Sample text for testing the SDPMChunker."""
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text


@pytest.fixture
def embedding_model():
    """Fixture that returns a SentenceTransformer embedding model for testing.

    Returns:
        SentenceTransformerEmbeddings: A sentence transformer model initialized with 'all-MiniLM-L6-v2'

    """
    return SentenceTransformerEmbeddings("all-MiniLM-L6-v2")


@pytest.fixture
def sample_complex_markdown_text():
    """Fixture that returns a sample markdown text with complex formatting.

    Returns:
        str: A markdown text containing various formatting elements like headings,
            lists, code blocks, links, images and blockquotes.

    """
    text = """# Heading 1
    This is a paragraph with some **bold text** and _italic text_. 
    ## Heading 2
    - Bullet point 1
    - Bullet point 2 with `inline code`
    ```python
    # Code block
    def hello_world():
        print("Hello, world!")
    ```
    Another paragraph with [a link](https://example.com) and an image:
    ![Alt text](https://example.com/image.jpg)
    > A blockquote with multiple lines
    > that spans more than one line.
    Finally, a paragraph at the end.
    """
    return text


def test_spdm_chunker_initialization(embedding_model):
    """Test that the SPDMChunker can be initialized with required parameters."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
        skip_window=2,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2
    assert chunker.skip_window == 2


def test_spdm_chunker_chunking(embedding_model, sample_text):
    """Test that the SPDMChunker can chunk a sample text."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], SemanticChunk)
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])
    assert all([chunk.sentences is not None for chunk in chunks])


def test_spdm_chunker_empty_text(embedding_model):
    """Test that the SPDMChunker can handle empty text input."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_spdm_chunker_single_sentence(embedding_model):
    """Test that the SPDMChunker can handle text with a single sentence."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("This is a single sentence.")

    assert len(chunks) == 1
    assert chunks[0].text == "This is a single sentence."
    assert len(chunks[0].sentences) == 1


def test_spdm_chunker_repr(embedding_model):
    """Test that the SPDMChunker has a string representation."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
        skip_window=2,
    )

    expected = (
        "SPDMChunker(embedding_model=SentenceTransformerEmbeddings(model=all-MiniLM-L6-v2), "
        "mode=window, threshold=0.5, chunk_size=512, similarity_window=1, "
        "min_sentences=1, min_chunk_size=2, min_characters_per_sentence=12, "
        "threshold_step=0.01, delim=['.', '!', '?', '\n'], skip_window=2)"
    )
    assert repr(chunker) == expected


def test_spdm_chunker_percentile_mode(embedding_model, sample_complex_markdown_text):
    """Test the SPDMChunker works with percentile-based similarity."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=50,
    )
    chunks = chunker.chunk(sample_complex_markdown_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], SemanticChunk)
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])
    assert all([chunk.sentences is not None for chunk in chunks])


def test_spdm_chunker_token_counts(embedding_model, sample_text):
    """Test that the SPDMChunker correctly calculates token counts."""
    chunker = SDPMChunker(embedding_model=embedding_model, chunk_size=512, threshold=0.5)
    chunks = chunker.chunk(sample_text)
    assert all([chunk.token_count > 0 for chunk in chunks]), "All chunks must have a positive token count"
    assert all([chunk.token_count <= 512 for chunk in chunks]), "All chunks must have a token count less than or equal to 512"  

    token_counts = [chunker._count_tokens(chunk.text) for chunk in chunks]
    assert all([chunk.token_count == token_count for chunk, token_count in zip(chunks, token_counts)]), "All chunks must have a token count equal to the length of the encoded text"

if __name__ == "__main__":
    pytest.main()
