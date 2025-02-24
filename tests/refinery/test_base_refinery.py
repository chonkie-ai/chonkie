"""Tests for the BaseRefinery class."""

import pytest

from chonkie.refinery.base import BaseRefinery
from chonkie.types import Chunk


# Create a concrete implementation of BaseRefinery for testing
class TestRefinery(BaseRefinery):

    """Test implementation of BaseRefinery."""

    def refine(self, chunks):
        """Test implementation of refine method."""
        return chunks

    @classmethod
    def is_available(cls):
        """Test implementation of is_available method."""
        return True


@pytest.fixture
def refinery():
    """Create a test refinery instance."""
    return TestRefinery(context_size=5)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(text="chunk1", token_count=10, start_index=0, end_index=10),
        Chunk(text="chunk2", token_count=15, start_index=10, end_index=25),
        Chunk(text="chunk3", token_count=20, start_index=25, end_index=45),
    ]


@pytest.fixture
def sample_chunks_batch():
    """Create sample batch of chunks for testing."""
    return [
        [
            Chunk(text="batch1_chunk1", token_count=10, start_index=0, end_index=10),
            Chunk(text="batch1_chunk2", token_count=15, start_index=10, end_index=25),
        ],
        [
            Chunk(text="batch2_chunk1", token_count=20, start_index=0, end_index=20),
            Chunk(text="batch2_chunk2", token_count=25, start_index=20, end_index=45),
        ],
    ]


def test_base_refinery_initialization():
    """Test BaseRefinery initialization."""
    refinery = TestRefinery(context_size=5)
    assert refinery.context_size == 5

    # Test negative context size
    with pytest.raises(ValueError):
        TestRefinery(context_size=-1)


def test_base_refinery_is_available():
    """Test is_available class method."""
    assert TestRefinery.is_available() is True


def test_base_refinery_refine(refinery, sample_chunks):
    """Test refine method."""
    refined_chunks = refinery.refine(sample_chunks)
    assert refined_chunks == sample_chunks
    assert all(isinstance(chunk, Chunk) for chunk in refined_chunks)


def test_base_refinery_refine_batch(refinery, sample_chunks_batch):
    """Test refine_batch method."""
    refined_batches = refinery.refine_batch(sample_chunks_batch)
    assert refined_batches == sample_chunks_batch
    assert all(isinstance(batch, list) for batch in refined_batches)
    assert all(isinstance(chunk, Chunk) for batch in refined_batches for chunk in batch)


def test_base_refinery_repr(refinery):
    """Test string representation."""
    assert str(refinery) == "TestRefinery(context_size=5)"
    assert repr(refinery) == "TestRefinery(context_size=5)"


def test_base_refinery_call_with_chunks(refinery, sample_chunks):
    """Test __call__ method with list of chunks."""
    refined_chunks = refinery(sample_chunks)
    assert refined_chunks == sample_chunks
    assert all(isinstance(chunk, Chunk) for chunk in refined_chunks)


def test_base_refinery_call_with_batch(refinery, sample_chunks_batch):
    """Test __call__ method with batch of chunks."""
    refined_batches = refinery(sample_chunks_batch)
    assert refined_batches == sample_chunks_batch
    assert all(isinstance(batch, list) for batch in refined_batches)
    assert all(isinstance(chunk, Chunk) for batch in refined_batches for chunk in batch)


def test_base_refinery_call_with_invalid_input(refinery):
    """Test __call__ method with invalid input."""
    # Test empty list
    assert refinery([]) == []

    # Test non-list input
    assert refinery(None) is None

    # Test list with invalid items
    with pytest.raises(ValueError):
        refinery([1, 2, 3])

    # Test list of lists with invalid items
    with pytest.raises(ValueError):
        refinery([[1, 2], [3, 4]])


def test_base_refinery_abstract_methods():
    """Test that abstract methods raise NotImplementedError."""

    class IncompleteRefinery(BaseRefinery):
        pass

    with pytest.raises(TypeError):
        IncompleteRefinery()
