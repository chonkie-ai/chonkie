"""Tests for Chonkie types."""

import numpy as np
import pytest

from chonkie.types import (
    Chunk,
    Context,
    RecursiveChunk,
    RecursiveLevel,
    RecursiveRules,
    SemanticSentence,
    Sentence,
    SentenceChunk,
)


# Context Tests
def test_context_init():
    """Test Context initialization."""
    context = Context(text="test", token_count=1)
    assert context.text == "test"
    assert context.token_count == 1
    assert context.start_index is None
    assert context.end_index is None


def test_context_validation():
    """Test Context validation."""
    # Test invalid text type
    with pytest.raises(ValueError):
        Context(text=123, token_count=1)

    # Test invalid token count type
    with pytest.raises(TypeError):
        Context(text="test", token_count="1")

    # Test negative token count
    with pytest.raises(ValueError):
        Context(text="test", token_count=-1)

    # Test invalid index range
    with pytest.raises(ValueError):
        Context(text="test", token_count=1, start_index=10, end_index=5)


def test_context_serialization():
    """Test Context serialization/deserialization."""
    context = Context(text="test", token_count=1, start_index=0, end_index=4)
    context_dict = context.to_dict()
    restored = Context.from_dict(context_dict)
    assert context.text == restored.text
    assert context.token_count == restored.token_count
    assert context.start_index == restored.start_index
    assert context.end_index == restored.end_index


# Chunk Tests
def test_chunk_init():
    """Test Chunk initialization."""
    chunk = Chunk(text="test chunk", start_index=0, end_index=10, token_count=2)
    assert chunk.text == "test chunk"
    assert chunk.start_index == 0
    assert chunk.end_index == 10
    assert chunk.token_count == 2
    assert chunk.context is None


def test_chunk_with_context():
    """Test Chunk with context."""
    context = Context(text="context", token_count=1)
    chunk = Chunk(
        text="test chunk", start_index=0, end_index=10, token_count=2, context=context
    )
    assert chunk.context == context


def test_chunk_serialization():
    """Test Chunk serialization/deserialization."""
    context = Context(text="context", token_count=1)
    chunk = Chunk(
        text="test chunk", start_index=0, end_index=10, token_count=2, context=context
    )
    chunk_dict = chunk.to_dict()
    restored = Chunk.from_dict(chunk_dict)
    assert chunk.text == restored.text
    assert chunk.token_count == restored.token_count
    assert chunk.context.text == restored.context.text


# Sentence Tests
def test_sentence_init():
    """Test Sentence initialization."""
    sentence = Sentence(
        text="test sentence.", start_index=0, end_index=14, token_count=3
    )
    assert sentence.text == "test sentence."
    assert sentence.start_index == 0
    assert sentence.end_index == 14
    assert sentence.token_count == 3


def test_sentence_serialization():
    """Test Sentence serialization/deserialization."""
    sentence = Sentence(
        text="test sentence.", start_index=0, end_index=14, token_count=3
    )
    sentence_dict = sentence.to_dict()
    restored = Sentence.from_dict(sentence_dict)
    assert sentence.text == restored.text
    assert sentence.token_count == restored.token_count


# SentenceChunk Tests
def test_sentence_chunk_init():
    """Test SentenceChunk initialization."""
    sentences = [
        Sentence("First sentence.", 0, 14, 3),
        Sentence("Second sentence.", 15, 30, 3),
    ]
    chunk = SentenceChunk(
        text="First sentence. Second sentence.",
        start_index=0,
        end_index=30,
        token_count=6,
        sentences=sentences,
    )
    assert chunk.text == "First sentence. Second sentence."
    assert len(chunk.sentences) == 2
    assert all(isinstance(s, Sentence) for s in chunk.sentences)


def test_sentence_chunk_serialization():
    """Test SentenceChunk serialization/deserialization."""
    sentences = [
        Sentence("First sentence.", 0, 14, 3),
        Sentence("Second sentence.", 15, 30, 3),
    ]
    chunk = SentenceChunk(
        text="First sentence. Second sentence.",
        start_index=0,
        end_index=30,
        token_count=6,
        sentences=sentences,
    )
    chunk_dict = chunk.to_dict()
    restored = SentenceChunk.from_dict(chunk_dict)
    assert len(restored.sentences) == 2
    assert all(isinstance(s, Sentence) for s in restored.sentences)


# SemanticSentence Tests
def test_semantic_sentence_init():
    """Test SemanticSentence initialization."""
    embedding = np.array([0.1, 0.2, 0.3])
    sentence = SemanticSentence(
        text="test sentence.",
        start_index=0,
        end_index=14,
        token_count=3,
        embedding=embedding,
    )
    assert sentence.text == "test sentence."
    assert np.array_equal(sentence.embedding, embedding)


def test_semantic_sentence_serialization():
    """Test SemanticSentence serialization/deserialization."""
    embedding = np.array([0.1, 0.2, 0.3])
    sentence = SemanticSentence(
        text="test sentence.",
        start_index=0,
        end_index=14,
        token_count=3,
        embedding=embedding,
    )
    sentence_dict = sentence.to_dict()
    restored = SemanticSentence.from_dict(sentence_dict)
    assert np.array_equal(restored.embedding, embedding)


# RecursiveLevel Tests
def test_recursive_level_init():
    """Test RecursiveLevel initialization."""
    level = RecursiveLevel(delimiters=["\n", "."])
    assert level.delimiters == ["\n", "."]
    assert not level.whitespace
    assert level.include_delim == "prev"


def test_recursive_level_validation():
    """Test RecursiveLevel validation."""
    with pytest.raises(ValueError):
        RecursiveLevel(delimiters=[1, 2])  # Invalid delimiter type

    with pytest.raises(ValueError):
        RecursiveLevel(delimiters=[""])  # Empty delimiter

    with pytest.raises(ValueError):
        RecursiveLevel(delimiters=[" "])  # Whitespace delimiter

    with pytest.raises(ValueError):
        RecursiveLevel(
            delimiters=["."], whitespace=True
        )  # Both delimiters and whitespace


def test_recursive_level_serialization():
    """Test RecursiveLevel serialization/deserialization."""
    level = RecursiveLevel(delimiters=["\n", "."])
    level_dict = level.to_dict()
    restored = RecursiveLevel.from_dict(level_dict)
    assert restored.delimiters == ["\n", "."]
    assert not restored.whitespace
    assert restored.include_delim == "prev"


# RecursiveRules Tests
def test_recursive_rules_default_init():
    """Test RecursiveRules default initialization."""
    rules = RecursiveRules()
    assert len(rules.levels) == 5
    assert all(isinstance(level, RecursiveLevel) for level in rules.levels)


def test_recursive_rules_custom_init():
    """Test RecursiveRules custom initialization."""
    levels = [
        RecursiveLevel(delimiters=["\n"]),
        RecursiveLevel(delimiters=["."]),
    ]
    rules = RecursiveRules(levels=levels)
    assert len(rules.levels) == 2
    assert rules.levels == levels


def test_recursive_rules_serialization():
    """Test RecursiveRules serialization/deserialization."""
    levels = [
        RecursiveLevel(delimiters=["\n"]),
        RecursiveLevel(delimiters=["."]),
    ]
    rules = RecursiveRules(levels=levels)
    rules_dict = rules.to_dict()
    restored = RecursiveRules.from_dict(rules_dict)
    assert len(restored.levels) == 2
    assert all(isinstance(level, RecursiveLevel) for level in restored.levels)


# RecursiveChunk Tests
def test_recursive_chunk_init():
    """Test RecursiveChunk initialization."""
    chunk = RecursiveChunk(
        text="test chunk", start_index=0, end_index=10, token_count=2, level=1
    )
    assert chunk.text == "test chunk"
    assert chunk.level == 1


def test_recursive_chunk_serialization():
    """Test RecursiveChunk serialization/deserialization."""
    chunk = RecursiveChunk(
        text="test chunk", start_index=0, end_index=10, token_count=2, level=1
    )
    chunk_dict = chunk.to_dict()
    restored = RecursiveChunk.from_dict(chunk_dict)
    assert restored.level == 1
    assert restored.text == chunk.text
