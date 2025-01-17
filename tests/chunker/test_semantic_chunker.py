import os
from typing import List

import pytest

from chonkie import SemanticChunker
from chonkie.embeddings import Model2VecEmbeddings, OpenAIEmbeddings
from chonkie.types import Chunk, SemanticChunk


@pytest.fixture
def sample_text():
    """Sample text for testing the SemanticChunker.

    Returns:
        str: A paragraph of text about text chunking in RAG applications.

    """
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text


@pytest.fixture
def embedding_model():
    """Fixture that returns a Model2Vec embedding model for testing.

    Returns:
        Model2VecEmbeddings: A Model2Vec model initialized with 'minishlab/potion-base-8M'

    """
    return Model2VecEmbeddings("minishlab/potion-base-8M")


@pytest.fixture
def openai_embedding_model():
    """Fixture that returns an OpenAI embedding model for testing.

    Returns:
        OpenAIEmbeddings: An OpenAI model initialized with 'text-embedding-3-small'
            and the API key from environment variables.

    """
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


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


def test_semantic_chunker_initialization(embedding_model):
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_semantic_chunker_initialization_openai(openai_embedding_model):
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        embedding_model=openai_embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


def test_semantic_chunker_initialization_sentence_transformer():
    """Test that the SemanticChunker can be initialized with SentenceTransformer model."""
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


def test_semantic_chunker_chunking(embedding_model, sample_text):
    """Test that the SemanticChunker can chunk a sample text."""
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
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


def test_semantic_chunker_empty_text(embedding_model):
    """Test that the SemanticChunker can handle empty text input."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_semantic_chunker_single_sentence(embedding_model):
    """Test that the SemanticChunker can handle text with a single sentence."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("This is a single sentence.")

    assert len(chunks) == 1
    assert chunks[0].text == "This is a single sentence."
    assert len(chunks[0].sentences) == 1


def test_semantic_chunker_repr(embedding_model):
    """Test that the SemanticChunker has a string representation."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    expected = (
        "SemanticChunker(embedding_model=Model2VecEmbeddings(model_name_or_path=minishlab/potion-base-8M), "
        "mode=window, chunk_size=512, threshold=0.5, similarity_window=1, "
        "min_sentences=1, min_chunk_size=2, min_characters_per_sentence=12, threshold_step=0.01, delim=['.', '!', '?', '\n'])"
    )
    assert repr(chunker) == expected


def test_semantic_chunker_similarity_threshold(embedding_model):
    """Test that the SemanticChunker respects similarity threshold."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.9,  # High threshold should create more chunks
    )
    text = (
        "This is about cars. This is about planes. "
        "This is about trains. This is about boats."
    )
    chunks = chunker.chunk(text)

    # With high similarity threshold, we expect more chunks due to low similarity
    assert len(chunks) > 1


def test_semantic_chunker_percentile_mode(embedding_model, sample_text):
    """Test that the SemanticChunker works with percentile-based similarity."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=50,  # Use median similarity as threshold
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert all([isinstance(chunk, SemanticChunk) for chunk in chunks])


def verify_chunk_indices(chunks: List[Chunk], original_text: str):
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


def test_sentence_chunker_indices(embedding_model, sample_text):
    """Test that the SentenceChunker correctly maps chunk indices to the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_sentence_chunker_indices_complex_md(
    embedding_model, sample_complex_markdown_text
):
    """Test that the SentenceChunker correctly maps chunk indices to the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_complex_markdown_text)
    verify_chunk_indices(chunks, sample_complex_markdown_text)

def test_semantic_chunker_token_counts(embedding_model, sample_text):
    """Test that the SemanticChunker correctly calculates token counts."""
    chunker = SemanticChunker(embedding_model=embedding_model, chunk_size=512, threshold=0.5)
    chunks = chunker.chunk(sample_text)
    assert all([chunk.token_count > 0 for chunk in chunks]), "All chunks must have a positive token count"
    assert all([chunk.token_count <= 512 for chunk in chunks]), "All chunks must have a token count less than or equal to 512"

    token_counts = [chunker._count_tokens(chunk.text) for chunk in chunks]
    for i, (chunk, token_count) in enumerate(zip(chunks, token_counts)):
        assert chunk.token_count == token_count, \
            f"Chunk {i} has a token count of {chunk.token_count} but the encoded text length is {token_count}"


def test_semantic_chunker_reconstruction(embedding_model, sample_text):
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SemanticChunker(embedding_model=embedding_model, chunk_size=512, threshold=0.5)
    chunks = chunker.chunk(sample_text)
    assert sample_text == "".join([chunk.text for chunk in chunks])


def test_semantic_chunker_reconstruction_complex_md(embedding_model, sample_complex_markdown_text):
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SemanticChunker(embedding_model=embedding_model, chunk_size=512, threshold=0.5)
    chunks = chunker.chunk(sample_complex_markdown_text)
    assert sample_complex_markdown_text == "".join([chunk.text for chunk in chunks])


def test_semantic_chunker_reconstruction_batch(embedding_model, sample_text):
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SemanticChunker(embedding_model=embedding_model, chunk_size=512, threshold=0.5)
    chunks = chunker.chunk_batch([sample_text]*10)[-1]
    assert sample_text == "".join([chunk.text for chunk in chunks])


if __name__ == "__main__":
    pytest.main()
