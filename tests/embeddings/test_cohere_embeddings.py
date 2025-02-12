import os

import numpy as np
import pytest

from chonkie.embeddings.cohere import CohereEmbeddings


@pytest.fixture
def embedding_model():
    api_key = os.environ.get("COHERE_API_KEY")
    return CohereEmbeddings(model="embed-english-light-v3.0", api_key=api_key)


@pytest.fixture
def sample_text():
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts():
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_initialization_with_model_name():
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    assert embeddings.model == "embed-english-light-v3.0"
    assert embeddings.client is not None


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_embed_single_text(embedding_model, sample_text):
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_embed_batch_texts(embedding_model, sample_texts):
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(
        embedding.shape == (embedding_model.dimension,) for embedding in embeddings
    )


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_count_tokens_single_text(embedding_model, sample_text):
    token_count = embedding_model.count_tokens(sample_text)
    assert isinstance(token_count, int)
    assert token_count > 0


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_count_tokens_batch_texts(embedding_model, sample_texts):
    token_counts = embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    assert all(isinstance(count, int) for count in token_counts)
    assert all(count > 0 for count in token_counts)


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_similarity(embedding_model, sample_texts):
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, float)
    assert 0.0 <= similarity_score <= 1.0


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_dimension_property(embedding_model):
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


def test_is_available():
    assert CohereEmbeddings.is_available() is True


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_repr(embedding_model):
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("CohereEmbeddings")


if __name__ == "__main__":
    pytest.main()
