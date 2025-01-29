"""Tests for the TokenChunker class.

This module contains test cases for the TokenChunker class, which implements
a token-based chunking strategy. The tests verify:

- Initialization with different tokenizer types
- Basic chunking functionality with simple and complex texts
- Batch processing capabilities
- Proper chunk generation and properties
- Handling of edge cases and different text formats

Fixtures:
- transformers_tokenizer: A GPT-2 tokenizer from the transformers library
- tiktokenizer: A GPT-2 tokenizer from the tiktoken library
- tokenizer: A GPT-2 tokenizer from the tokenizers library
- sample_text: A sample text for testing chunking
- sample_batch: A batch of texts from a dataset for testing batch processing
- sample_complex_markdown_text: A markdown text with various formatting elements

"""

from typing import List

import pytest
import tiktoken
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from chonkie import Chunk, TokenChunker


@pytest.fixture
def transformers_tokenizer():
    """Fixture that returns a GPT-2 tokenizer from the transformers library.

    Returns:
        AutoTokenizer: A GPT-2 tokenizer initialized from the 'gpt2' model

    """
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def tiktokenizer():
    """Fixture that returns a GPT-2 tokenizer from the tiktoken library.

    Returns:
        tiktoken.Encoding: A GPT-2 tokenizer initialized from the 'gpt2' encoding

    """
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def tokenizer():
    """Fixture that returns a GPT-2 tokenizer from the tokenizers library.

    Returns:
        Tokenizer: A GPT-2 tokenizer initialized from the 'gpt2' model

    """
    return Tokenizer.from_pretrained("gpt2")


@pytest.fixture
def sample_text():
    """Fixture that returns a sample text for testing the TokenChunker.

    Returns:
        str: A sample text containing multiple sentences discussing text chunking
            in RAG applications.

    """
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text


@pytest.fixture
def sample_batch():
    """Fixture that returns a sample batch of texts for testing the TokenChunker.

    This fixture loads a small dataset of educational texts from the Hugging Face
    datasets library. It's useful for testing batch processing capabilities and
    performance with real-world data.

    Returns:
        List[str]: A list of text samples from the dataset.

    """
    ds = load_dataset("bhavnicksm/fineweb-edu-micro", split="train")
    return list(ds["text"])


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


def test_token_chunker_initialization_tok(tokenizer):
    """Test that the TokenChunker can be initialized with a tokenizer."""
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)

    assert chunker is not None
    assert chunker.tokenizer == tokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128


def test_token_chunker_initialization_hftok(transformers_tokenizer):
    """Test that the TokenChunker can be initialized with a tokenizer."""
    chunker = TokenChunker(
        tokenizer=transformers_tokenizer, chunk_size=512, chunk_overlap=128
    )

    assert chunker is not None
    assert chunker.tokenizer == transformers_tokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128


def test_token_chunker_initialization_tik(tiktokenizer):
    """Test that the TokenChunker can be initialized with a tokenizer."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)

    assert chunker is not None
    assert chunker.tokenizer == tiktokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128


def test_token_chunker_chunking(tiktokenizer, sample_text):
    """Test that the TokenChunker can chunk a sample text into tokens."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_token_chunker_chunking_hf(transformers_tokenizer, sample_text):
    """Test that the TokenChunker can chunk a sample text into tokens."""
    chunker = TokenChunker(
        tokenizer=transformers_tokenizer, chunk_size=512, chunk_overlap=128
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_token_chunker_chunking_tik(tiktokenizer, sample_text):
    """Test that the TokenChunker can chunk a sample text into tokens."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_token_chunker_empty_text(tiktokenizer):
    """Test that the TokenChunker can handle empty text input."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_token_chunker_single_token_text(tokenizer):
    """Test that the TokenChunker can handle text with a single token."""
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello")

    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].text == "Hello"


def test_token_chunker_single_token_text_hf(transformers_tokenizer):
    """Test that the TokenChunker can handle text with a single token."""
    chunker = TokenChunker(
        tokenizer=transformers_tokenizer, chunk_size=512, chunk_overlap=128
    )
    chunks = chunker.chunk("Hello")

    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].text == "Hello"


def test_token_chunker_single_token_text_tik(tiktokenizer):
    """Test that the TokenChunker can handle text with a single token."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello")

    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].text == "Hello"


def test_token_chunker_single_chunk_text(tokenizer):
    """Test that the TokenChunker can handle text that fits within a single chunk."""
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello, how are you?")

    assert len(chunks) == 1
    assert chunks[0].token_count == 6
    assert chunks[0].text == "Hello, how are you?"


def test_token_chunker_batch_chunking(tiktokenizer, sample_batch):
    """Test that the TokenChunker can chunk a batch of texts into tokens."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk_batch(sample_batch)

    assert len(chunks) > 0
    assert all([len(chunk) > 0 for chunk in chunks])
    assert all([type(chunk[0]) is Chunk for chunk in chunks])
    assert all(
        [all([chunk.token_count <= 512 for chunk in chunks]) for chunks in chunks]
    )
    assert all([all([chunk.token_count > 0 for chunk in chunks]) for chunks in chunks])
    assert all([all([chunk.text is not None for chunk in chunks]) for chunks in chunks])
    assert all(
        [all([chunk.start_index is not None for chunk in chunks]) for chunks in chunks]
    )
    assert all(
        [all([chunk.end_index is not None for chunk in chunks]) for chunks in chunks]
    )


def test_token_chunker_repr(tiktokenizer):
    """Test that the TokenChunker has a string representation."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)

    assert repr(chunker) == "TokenChunker(tokenizer=<Encoding 'gpt2'>, chunk_size=512, chunk_overlap=128)"


def test_token_chunker_call(tiktokenizer, sample_text):
    """Test that the TokenChunker can be called directly."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


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


def test_token_chunker_indices(tiktokenizer, sample_text):
    """Test that TokenChunker's indices correctly map to original text."""
    tokenizer = Tokenizer.from_pretrained("gpt2")
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_token_chunker_indices_complex_md(sample_complex_markdown_text):
    """Test that TokenChunker's indices correctly map to original text."""
    tokenizer = Tokenizer.from_pretrained("gpt2")
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_complex_markdown_text)
    verify_chunk_indices(chunks, sample_complex_markdown_text)


def test_token_chunker_token_counts(tiktokenizer, sample_text):
    """Test that the TokenChunker correctly calculates token counts."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)
    assert all([chunk.token_count > 0 for chunk in chunks]), "All chunks must have a positive token count"
    assert all([chunk.token_count <= 512 for chunk in chunks]), "All chunks must have a token count less than or equal to 512"  

    token_counts = [len(tiktokenizer.encode(chunk.text)) for chunk in chunks]
    assert all([chunk.token_count == token_count for chunk, token_count in zip(chunks, token_counts)]), "All chunks must have a token count equal to the length of the encoded text"

def test_token_chunker_indices_batch(tiktokenizer, sample_text):
    """Test that TokenChunker's indices correctly map to original text."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk_batch([sample_text]*10)[-1]
    verify_chunk_indices(chunks, sample_text)

def test_token_chunker_return_type(tiktokenizer, sample_text):
    """Test that TokenChunker's return type is correctly set."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128, return_type="texts")
    chunks = chunker.chunk(sample_text)
    assert all([type(chunk) is str for chunk in chunks])
    assert all([len(tiktokenizer.encode(chunk)) <= 512 for chunk in chunks])

if __name__ == "__main__":
    pytest.main()
