import pytest
from typing import List
from chonkie.chunker.base import Chunk
from chonkie.chunker.sentence import SentenceChunker
from chonkie.chunker.word import WordChunker
from chonkie.chunker.token import TokenChunker
from chonkie.chunker.sdpm import SDPMChunker
from chonkie.chunker.semantic import SemanticChunker
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from tokenizers import Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")


@pytest.fixture
def embedding_model():
    return SentenceTransformerEmbeddings("all-MiniLM-L6-v2")


@pytest.fixture
def complex_markdown_text():
    return """
# Heading 1

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

def verify_chunk_indices(chunks: List[Chunk], original_text: str):
    """Verify that chunk indices correctly map to the original text."""
    for i, chunk in enumerate(chunks):
        extracted_text = original_text[chunk.start_index:chunk.end_index]
        chunk_text = chunk.text.strip()
        extracted_text = extracted_text.strip()
        assert chunk_text == extracted_text, (
            f"Chunk {i} text mismatch:\n"
            f"Chunk text: '{chunk_text}'\n"
            f"Extracted text: '{extracted_text}'\n"
            f"Indices: [{chunk.start_index}:{chunk.end_index}]"
        )

def test_sentence_chunker(tokenizer, complex_markdown_text):
    """Test SentenceChunker chunking and reconstruction."""
    chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(complex_markdown_text)
    reconstructed_text = "".join(complex_markdown_text[chunk.start_index:chunk.end_index] for chunk in chunks)
    assert complex_markdown_text.strip() == reconstructed_text.strip()
    verify_chunk_indices(chunks, complex_markdown_text)


def test_word_chunker(tokenizer, complex_markdown_text):
    """Test WordChunker chunking and reconstruction."""
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(complex_markdown_text)
    reconstructed_text = "".join(complex_markdown_text[chunk.start_index:chunk.end_index] for chunk in chunks)
    assert complex_markdown_text.strip() == reconstructed_text.strip()
    verify_chunk_indices(chunks, complex_markdown_text)


def test_token_chunker(tokenizer, complex_markdown_text):
    """Test TokenChunker chunking and reconstruction."""
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(complex_markdown_text)
    reconstructed_text = "".join(complex_markdown_text[chunk.start_index:chunk.end_index] for chunk in chunks)
    assert complex_markdown_text.strip() == reconstructed_text.strip()
    verify_chunk_indices(chunks, complex_markdown_text)


def test_sdpm_chunker(embedding_model, complex_markdown_text):
    """Test SDPMChunker chunking and reconstruction."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        max_chunk_size=512,
        similarity_threshold=0.5,
        skip_window=2,
    )
    chunks = chunker.chunk(complex_markdown_text)
    reconstructed_text = "".join(complex_markdown_text[chunk.start_index:chunk.end_index] for chunk in chunks)
    assert complex_markdown_text.strip() == reconstructed_text.strip()
    verify_chunk_indices(chunks, complex_markdown_text)


def test_semantic_chunker(embedding_model, complex_markdown_text):
    """Test SemanticChunker chunking and reconstruction."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        max_chunk_size=512,
        similarity_threshold=0.5,
    )
    chunks = chunker.chunk(complex_markdown_text)
    reconstructed_text = "".join(complex_markdown_text[chunk.start_index:chunk.end_index] for chunk in chunks)
    assert complex_markdown_text.strip() == reconstructed_text.strip()
    verify_chunk_indices(chunks, complex_markdown_text)


if __name__ == "__main__":
    pytest.main()
