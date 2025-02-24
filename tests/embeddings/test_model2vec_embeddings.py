"""Test the Model2VecEmbeddings class."""

import numpy as np
import pytest
from model2vec import StaticModel

from chonkie import Model2VecEmbeddings


@pytest.fixture
def embedding_model():
    """Return a Model2VecEmbeddings instance."""
    return Model2VecEmbeddings("minishlab/potion-base-8M")


@pytest.fixture
def sample_text():
    """Return a sample text for testing."""
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts():
    """Return a list of sample texts for testing."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


def test_initialization_with_model_name():
    """Test that the Model2VecEmbeddings instance is initialized correctly with a model name."""
    embeddings = Model2VecEmbeddings("minishlab/potion-base-8M")
    assert (
        embeddings.model_name_or_path == "minishlab/potion-base-8M"
    )  # for now its None, see comments in model_2_vec.py
    assert embeddings.model is not None


def test_initialization_with_model_instance():
    """Test that the Model2VecEmbeddings instance is initialized correctly with a model instance."""
    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    embeddings = Model2VecEmbeddings(model)
    assert embeddings.model_name_or_path == model.base_model_name
    assert embeddings.model is model


def test_embed_single_text(embedding_model, sample_text):
    """Test that the embed method returns a numpy array of the correct shape."""
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


def test_count_tokens_batch_texts(embedding_model, sample_texts):
    """Test that the count_tokens_batch method returns a list of token counts."""
    token_counts = embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    assert all(isinstance(count, int) for count in token_counts)
    assert all(count > 0 for count in token_counts)


def test_similarity(embedding_model, sample_texts):
    """Test that the similarity method returns a float between 0 and 1."""
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, np.float32), (
        f"Similarity score is not a float: {type(similarity_score)}"
    )
    assert 0.0 <= similarity_score <= 1.0, (
        f"Similarity score is not between 0 and 1: {similarity_score}"
    )


def test_dimension_property(embedding_model):
    """Test that the dimension property returns an integer greater than 0."""
    assert isinstance(embedding_model.dimension, int), (
        f"Dimension is not an integer: {type(embedding_model.dimension)}"
    )
    assert embedding_model.dimension > 0, (
        f"Dimension is not greater than 0: {embedding_model.dimension}"
    )


def test_is_available():
    """Test that the is_available method returns True."""
    assert Model2VecEmbeddings.is_available() is True


def test_repr(embedding_model):
    """Test that the repr method returns a string starting with 'Model2VecEmbeddings'."""
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str), f"repr_str is not a string: {type(repr_str)}"
    assert repr_str.startswith("Model2VecEmbeddings"), (
        f"repr_str does not start with 'Model2VecEmbeddings': {repr_str}"
    )


if __name__ == "__main__":
    pytest.main()
