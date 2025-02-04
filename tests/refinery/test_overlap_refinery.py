from typing import List

import pytest
from transformers import AutoTokenizer

from chonkie import TokenChunker
from chonkie.refinery import OverlapRefinery
from chonkie.types import Chunk, Context, Sentence, SentenceChunk, RecursiveRules, RecursiveLevel


@pytest.fixture
def tokenizer():
    """Fixture providing a GPT-2 tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def basic_chunks() -> List[Chunk]:
    """Fixture providing a list of basic Chunks for testing."""
    return [
        Chunk(
            text="This is the first chunk of text.",
            start_index=0,
            end_index=30,
            token_count=8,
        ),
        Chunk(
            text="This is the second chunk of text.",
            start_index=31,
            end_index=62,
            token_count=8,
        ),
        Chunk(
            text="This is the third chunk of text.",
            start_index=63,
            end_index=93,
            token_count=8,
        ),
    ]


@pytest.fixture
def sentence_chunks() -> List[SentenceChunk]:
    """Fixture providing a list of SentenceChunks for testing."""
    sentences1 = [
        Sentence(text="First sentence.", start_index=0, end_index=14, token_count=3),
        Sentence(text="Second sentence.", start_index=15, end_index=30, token_count=3),
    ]
    sentences2 = [
        Sentence(text="Third sentence.", start_index=31, end_index=45, token_count=3),
        Sentence(text="Fourth sentence.", start_index=46, end_index=62, token_count=3),
    ]
    return [
        SentenceChunk(
            text="First sentence. Second sentence.",
            start_index=0,
            end_index=30,
            token_count=6,
            sentences=sentences1,
        ),
        SentenceChunk(
            text="Third sentence. Fourth sentence.",
            start_index=31,
            end_index=62,
            token_count=6,
            sentences=sentences2,
        ),
    ]

@pytest.fixture
def sample_text():
    """Return a sample text."""
    text = """# Chunking Strategies in Retrieval-Augmented Generation: A Comprehensive Analysis\n\nIn the rapidly evolving landscape of natural language processing, Retrieval-Augmented Generation (RAG) has emerged as a groundbreaking approach that bridges the gap between large language models and external knowledge bases. At the heart of these systems lies a crucial yet often overlooked process: chunking. This fundamental operation, which involves the systematic decomposition of large text documents into smaller, semantically meaningful units, plays a pivotal role in determining the overall effectiveness of RAG implementations.\n\nThe process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. This balancing act becomes particularly crucial when we consider the downstream implications for vector databases and embedding models that form the backbone of modern RAG systems.\n\nThe selection of appropriate chunk size emerges as a fundamental consideration that significantly impacts system performance. Through extensive experimentation and real-world implementations, researchers have identified that chunks typically perform optimally in the range of 256 to 1024 tokens. However, this range should not be treated as a rigid constraint but rather as a starting point for optimization based on specific use cases and requirements. The implications of chunk size selection ripple throughout the entire RAG pipeline, affecting everything from storage requirements to retrieval accuracy and computational overhead.\n\nFixed-size chunking represents the most straightforward approach to document segmentation, offering predictable memory usage and consistent processing time. However, this apparent simplicity comes with significant drawbacks. By arbitrarily dividing text based on token or character count, fixed-size chunking risks fragmenting semantic units and disrupting the natural flow of information. Consider, for instance, a technical document where a complex concept is explained across several paragraphs – fixed-size chunking might split this explanation at critical junctures, potentially compromising the system's ability to retrieve and present this information coherently.\n\nIn response to these limitations, semantic chunking has gained prominence as a more sophisticated alternative. This approach leverages natural language understanding to identify meaningful boundaries within the text, respecting the natural structure of the document. Semantic chunking can operate at various levels of granularity, from simple sentence-based segmentation to more complex paragraph-level or topic-based approaches. The key advantage lies in its ability to preserve the inherent semantic relationships within the text, leading to more meaningful and contextually relevant retrieval results.\n\nRecent advances in the field have given rise to hybrid approaches that attempt to combine the best aspects of both fixed-size and semantic chunking. These methods typically begin with semantic segmentation but impose size constraints to prevent extreme variations in chunk length. Furthermore, the introduction of sliding window techniques with overlap has proved particularly effective in maintaining context across chunk boundaries. This overlap, typically ranging from 10% to 20% of the chunk size, helps ensure that no critical information is lost at segment boundaries, albeit at the cost of increased storage requirements.\n\nThe implementation of chunking strategies must also consider various technical factors that can significantly impact system performance. Vector database capabilities, embedding model constraints, and runtime performance requirements all play crucial roles in determining the optimal chunking approach. Moreover, content-specific factors such as document structure, language characteristics, and domain-specific requirements must be carefully considered. For instance, technical documentation might benefit from larger chunks that preserve detailed explanations, while news articles might perform better with smaller, more focused segments.\n\nThe future of chunking in RAG systems points toward increasingly sophisticated approaches. Current research explores the potential of neural chunking models that can learn optimal segmentation strategies from large-scale datasets. These models show promise in adapting to different content types and query patterns, potentially leading to more efficient and effective retrieval systems. Additionally, the emergence of cross-lingual chunking strategies addresses the growing need for multilingual RAG applications, while real-time adaptive chunking systems attempt to optimize segment boundaries based on user interaction patterns and retrieval performance metrics.\n\nThe effectiveness of RAG systems heavily depends on the thoughtful implementation of appropriate chunking strategies. While the field continues to evolve, practitioners must carefully consider their specific use cases and requirements when designing chunking solutions. Factors such as document characteristics, retrieval patterns, and performance requirements should guide the selection and optimization of chunking strategies. As we look to the future, the continued development of more sophisticated chunking approaches promises to further enhance the capabilities of RAG systems, enabling more accurate and efficient information retrieval and generation.\n\nThrough careful consideration of these various aspects and continued experimentation with different approaches, organizations can develop chunking strategies that effectively balance the competing demands of semantic coherence, computational efficiency, and retrieval accuracy. As the field continues to evolve, we can expect to see new innovations that further refine our ability to segment and process textual information in ways that enhance the capabilities of RAG systems while maintaining their practical utility in real-world applications."""
    return text

def test_overlap_refinery_initialization():
    """Test that OverlapRefinery initializes correctly with different parameters."""
    # Test default initialization
    refinery = OverlapRefinery()
    assert refinery.context_size == 128
    assert refinery.merge_context is True
    assert refinery.approximate is True
    assert not hasattr(refinery, "tokenizer")

    # Test initialization with tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    refinery = OverlapRefinery(
        context_size=64, tokenizer=tokenizer, merge_context=False, approximate=False
    )
    assert refinery.context_size == 64
    assert refinery.merge_context is False
    assert refinery.approximate is False
    assert hasattr(refinery, "tokenizer")
    assert refinery.tokenizer == tokenizer


def test_overlap_refinery_empty_input():
    """Test that OverlapRefinery handles empty input correctly."""
    refinery = OverlapRefinery()
    assert refinery.refine([]) == []


def test_overlap_refinery_single_chunk():
    """Test that OverlapRefinery handles single chunk input correctly."""
    refinery = OverlapRefinery()
    chunk = Chunk(text="Single chunk.", start_index=0, end_index=12, token_count=3)
    refined = refinery.refine([chunk])
    assert len(refined) == 1
    assert refined[0].context is None


def test_overlap_refinery_basic_chunks_approximate(basic_chunks):
    """Test approximate overlap calculation with basic Chunks."""
    refinery = OverlapRefinery(context_size=4)  # Small context for testing
    refined = refinery.refine(basic_chunks)

    # Last chunk should have no context
    assert refined[-1].context is None

    # Subsequent chunks should have context from previous chunks
    for i in range(len(refined) - 1):
        assert refined[i].context is not None
        assert isinstance(refined[i].context, Context)
        assert refined[i].context.token_count <= 4


def test_overlap_refinery_basic_chunks_exact(basic_chunks, tokenizer):
    """Test exact overlap calculation with basic Chunks using tokenizer."""
    refinery = OverlapRefinery(context_size=4, tokenizer=tokenizer, approximate=False)
    refined = refinery.refine(basic_chunks)

    # Check context for subsequent chunks
    for i in range(len(refined) - 1):
        assert refined[i].context is not None
        assert isinstance(refined[i].context, Context)
        # Verify exact token count using tokenizer
        actual_tokens = len(tokenizer.encode(refined[i].context.text))
        assert actual_tokens <= 4


def test_overlap_refinery_sentence_chunks(sentence_chunks):
    """Test overlap calculation with SentenceChunks."""
    refinery = OverlapRefinery(context_size=4)
    refined = refinery.refine(sentence_chunks)

    # Check context for first chunk
    assert refined[0].context is not None
    assert isinstance(refined[0].context, Context)


def test_overlap_refinery_no_merge_context(basic_chunks):
    """Test behavior when merge_context is False."""
    refinery = OverlapRefinery(context_size=4, merge_context=False)
    refined = refinery.refine(basic_chunks)

    # Chunks should maintain original text
    for i in range(len(refined)):
        assert refined[i].text == basic_chunks[i].text
        assert refined[i].token_count == basic_chunks[i].token_count


def test_overlap_refinery_context_size_limits(basic_chunks):
    """Test that context size limits are respected."""
    refinery = OverlapRefinery(context_size=2)  # Very small context
    refined = refinery.refine(basic_chunks)

    # Check that no context exceeds size limit
    for chunk in refined[:-1]:  # Skip last chunk
        assert chunk.context.token_count <= 2


def test_overlap_refinery_merge_context(basic_chunks, tokenizer):
    """Test merging context into chunk text."""
    refinery = OverlapRefinery(
        context_size=4, tokenizer=tokenizer, merge_context=True, approximate=False
    )

    # Create a deep copy to preserve originals
    chunks_copy = [
        Chunk(
            text=chunk.text,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            token_count=chunk.token_count,
        )
        for chunk in basic_chunks
    ]

    refined = refinery.refine(chunks_copy)

    # Subsequent chunks should have context prepended
    for i in range(len(refined) - 1):
        assert refined[i].context is not None
        assert refined[i].text.endswith(refined[i].context.text)
        # Verify token count increase
        original_tokens = len(tokenizer.encode(basic_chunks[i].text))
        new_tokens = len(tokenizer.encode(refined[i].text))
        assert new_tokens > original_tokens


def test_overlap_refinery_mixed_chunk_types():
    """Test that refinery raises error for mixed chunk types."""
    # Create chunks of different types
    chunks = [
        Chunk(text="Basic chunk.", start_index=0, end_index=12, token_count=3),
        SentenceChunk(
            text="Sentence chunk.",
            start_index=13,
            end_index=27,
            token_count=3,
            sentences=[],
        ),
    ]

    refinery = OverlapRefinery()
    with pytest.raises(ValueError, match="All chunks must be of the same type"):
        refinery.refine(chunks)


def test_overlap_refinery_prefix_mode(basic_chunks):
    """Test that OverlapRefinery works correctly in prefix mode."""
    refinery = OverlapRefinery(context_size=4, mode="prefix")
    refined = refinery.refine(basic_chunks)

    # First chunk should have no context
    assert refined[0].context is None

    # Subsequent chunks should have context from previous chunks
    for i in range(1, len(refined)):
        assert refined[i].context is not None
        assert isinstance(refined[i].context, Context)
        assert refined[i].context.token_count <= 4
        # Verify context comes from previous chunk
        assert refined[i].context.text in basic_chunks[i - 1].text
        # Verify context ends at the end of previous chunk
        assert refined[i].context.end_index == basic_chunks[i - 1].end_index


def test_overlap_refinery_prefix_mode_with_merge(basic_chunks, tokenizer):
    """Test that OverlapRefinery merges context correctly in prefix mode."""
    refinery = OverlapRefinery(
        context_size=4,
        tokenizer=tokenizer,
        mode="prefix",
        merge_context=True,
        approximate=False,
    )
    refined = refinery.refine(basic_chunks)

    # First chunk should be unchanged
    assert refined[0].text == basic_chunks[0].text
    assert refined[0].token_count == basic_chunks[0].token_count

    # Subsequent chunks should have context prepended
    for i in range(1, len(refined)):
        assert refined[i].context is not None
        # Verify text starts with context
        assert refined[i].text.startswith(refined[i].context.text)
        # Verify token count increase
        original_tokens = len(tokenizer.encode(basic_chunks[i].text))
        new_tokens = len(tokenizer.encode(refined[i].text))
        assert (
            new_tokens >= original_tokens
        ), f"{refined[i].text} {basic_chunks[i].text}"
        # Verify start index is from context
        assert refined[i].start_index == refined[i].context.start_index


def test_overlap_refinery_sample_text_suffix(sample_text):
    """Test that OverlapRefinery works correctly with a sample text in suffix mode."""
    # Get chunks from sample text
    chunker = TokenChunker(chunk_size=384, chunk_overlap=0)
    tokenizer = chunker.tokenizer
    chunks = chunker.chunk(sample_text)
    # Initialize refinery and refine chunks
    refinery = OverlapRefinery(context_size=128,
                               tokenizer=tokenizer,
                               mode="suffix",
                               merge_context=True, 
                               approximate=False)
    refined = refinery.refine(chunks)

    # Check the size of the refined chunks
    assert len(refined) == len(chunks)
    for i, chunk in enumerate(refined):
        if i != len(refined) - 1:
            assert chunker._count_tokens(chunk.text) == 512, \
                f"Chunk {i} has {chunker._count_tokens(chunk.text)} tokens"
        else:
            assert chunker._count_tokens(chunk.text) == chunk.token_count, \
                f"Chunk {i} has {chunker._count_tokens(chunk.text)} tokens"


def test_overlap_refinery_sample_text_prefix(sample_text):
    """Test that OverlapRefinery works correctly with a sample text in prefix mode."""
    # Get chunks from sample text
    chunker = TokenChunker(chunk_size=384, chunk_overlap=0)
    tokenizer = chunker.tokenizer
    chunks = chunker.chunk(sample_text)
    original_token_count = [chunker._count_tokens(chunk.text) for chunk in chunks]

    # Initialize refinery and refine chunks
    refinery = OverlapRefinery(context_size=128,
                               tokenizer=tokenizer,
                               mode="prefix",
                               merge_context=True, 
                               approximate=False)
    refined = refinery.refine(chunks)

    # Check the size of the refined chunks
    assert len(refined) == len(chunks)
    
    actual_token_count = []
    predicted_token_count = []
    for i, chunk in enumerate(refined):
        if i != 0:
            actual_token_count.append(chunker._count_tokens(chunk.text))
            predicted_token_count.append(original_token_count[i] + 128)
        else:
            actual_token_count.append(chunk.token_count)
            predicted_token_count.append(chunk.token_count)

    assert actual_token_count == predicted_token_count, \
        f"Actual token count: {actual_token_count} does not match predicted token count: {predicted_token_count}"

# Hierarchical Functions
@pytest.fixture
def recursive_rules():
    """Fixture providing sample recursive rules for testing."""

    # First level: Split by speaker changes (double newline)
    speaker_level = RecursiveLevel(
        delimiters=["\n\n"],
        whitespace=False
    )

    word_level = RecursiveLevel(
        delimiters=None,
        whitespace=True
    )

    token_level = RecursiveLevel(
        delimiters=None,
        whitespace=False
    )
    
    return RecursiveRules(levels=[
        speaker_level,
        word_level,
        token_level
    ])


@pytest.fixture
def hierarchical_text():
    """Fixture providing sample text with clear hierarchical structure."""
    return """Chapter 1: Introduction

This is the first paragraph of the introduction.
It has multiple sentences and spans multiple lines.

This is the second paragraph.
Another multi-line structure.

Chapter 2: Methods

The methods section begins here.
It contains important information."""

@pytest.fixture
def hierarchical_chunks(hierarchical_text):
    """Fixture providing pre-chunked hierarchical text."""
    chunks = [
        Chunk(
            text="Chapter 1: Introduction\n\nThis is the first paragraph",
            start_index=0,
            end_index=50,
            token_count=12
        ),
        Chunk(
            text="of the introduction.\nIt has multiple sentences",
            start_index=51,
            end_index=100,
            token_count=10
        ),
        Chunk(
            text="and spans multiple lines.\n\nThis is the second paragraph.",
            start_index=101,
            end_index=160,
            token_count=14
        )
    ]
    return chunks

def test_recursive_refinery_initialization():
    """Test initialization of OverlapRefinery in recursive mode."""
    # Test with minimum required parameters
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=64
    )
    assert refinery.method == "recursive"
    assert refinery.context_size == 128
    assert refinery.min_tokens == 64
    
    # Test initialization with invalid parameters
    with pytest.raises(ValueError, match="min_tokens must be specified"):
        OverlapRefinery(method="recursive", context_size=128)
        
    with pytest.raises(ValueError, match="min_tokens cannot be greater than context_size"):
        OverlapRefinery(method="recursive", context_size=64, min_tokens=128)

def test_recursive_refinery_with_rules(recursive_rules, hierarchical_chunks):
    """Test recursive refinery with custom rules."""
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=64,
        rules=recursive_rules
    )
    refined = refinery.refine(hierarchical_chunks)
    
    # Verify refinement results
    assert len(refined) == len(hierarchical_chunks)
    
    # Check that subsequent chunks have context
    for i in range(1, (len(refined) - 1)):
        assert refined[i].context is not None
        assert refined[i].context.token_count <= 128

def test_recursive_refinery_boundary_detection(hierarchical_text, recursive_rules):
    """Test that recursive refinery correctly identifies boundaries."""
    rules = recursive_rules # Only paragraph breaks
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=64,
        rules=rules
    )
    
    chunks = [
        Chunk(
            text=hierarchical_text,
            start_index=0,
            end_index=len(hierarchical_text),
            token_count=50
        )
    ]
    
    refined = refinery.refine(chunks)
    
    # Verify that paragraph breaks are respected
    if refined[0].context:
        assert "\n\n" in refined[0].context.text

def test_recursive_refinery_whitespace_fallback(recursive_rules):
    """Test fallback to whitespace splitting when no delimiters match."""
    # Text without explicit delimiters
    text = "This is a long sentence without any special delimiters that could be used for splitting into meaningful chunks of text"
    
    rules = recursive_rules
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=32,
        rules=rules
    )
    
    chunks = [
        Chunk(
            text=text,
            start_index=0,
            end_index=len(text),
            token_count=40
        )
    ]
    
    refined = refinery.refine(chunks)
    assert len(refined) == len(chunks)
    if refined[0].context:
        # Should have found some whitespace-based split
        assert " " in refined[0].context.text

def test_recursive_refinery_with_small_chunk(hierarchical_chunks, recursive_rules):
    """Test recursive refinery returns whole chunk if it's less than the minimum specified."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    rules = recursive_rules
    
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=64,
        rules=rules,
        tokenizer=tokenizer,
        approximate=False
    )
    
    refined = refinery.refine(hierarchical_chunks)
    
    # Verify exact token counts
    for chunk in refined[:-1]:  # Skip last chunk
        if chunk.context:
            actual_tokens = len(tokenizer.encode(chunk.context.text))
            print(actual_tokens)
            assert chunk.context.token_count == actual_tokens

def test_recursive_refinery_suffix_mode(hierarchical_chunks, recursive_rules):
    """Test recursive refinery in suffix mode."""
    rules = recursive_rules
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=64,
        rules=rules,
        mode="suffix"
    )
    
    refined = refinery.refine(hierarchical_chunks)
    
    # Check suffix context
    for chunk in refined[:-1]:  # Skip last chunk
        if chunk.context:
            # Context should come from next chunk
            next_chunk_index = hierarchical_chunks.index(chunk) + 1
            assert chunk.context.text in hierarchical_chunks[next_chunk_index].text

def test_recursive_refinery_merge_context(hierarchical_chunks, recursive_rules):
    """Test context merging in recursive mode."""
    rules = recursive_rules
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=64,
        rules=rules,
        merge_context=True,
        inplace=False
    )
    
    refined = refinery.refine(hierarchical_chunks)
    
    # Verify merged context
    for i in range(1, len(refined)):
        if refined[i].context:
            # Text should include context
            assert refined[i].text.endswith(refined[i].context.text)
            # Token count should be increased
            print(refined[i])
            assert refined[i].token_count > hierarchical_chunks[i].token_count

def test_recursive_refinery_empty_input():
    """Test recursive refinery with empty input."""
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=64
    )
    assert refinery.refine([]) == []

def test_recursive_refinery_single_chunk():
    """Test recursive refinery with a single chunk."""
    chunk = Chunk(
        text="Single chunk of text.",
        start_index=0,
        end_index=19,
        token_count=5
    )
    
    refinery = OverlapRefinery(
        method="recursive",
        context_size=128,
        min_tokens=64
    )
    
    refined = refinery.refine([chunk])
    assert len(refined) == 1
    assert refined[0].context is None


if __name__ == "__main__":
    pytest.main()