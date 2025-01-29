"""Tests for the AutoEmbeddings class."""

import pytest

from chonkie import AutoEmbeddings
from chonkie.embeddings.model2vec import Model2VecEmbeddings
from chonkie.embeddings.openai import OpenAIEmbeddings
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from chonkie.embeddings.litellm import LiteLLMEmbeddings

@pytest.fixture
def model2vec_identifier():
    """Fixture providing a model2vec identifier."""
    return "minishlab/potion-base-8M"


@pytest.fixture
def sentence_transformer_identifier():
    """Fixture providing a sentence transformer identifier."""
    return "all-MiniLM-L6-v2"


@pytest.fixture
def sentence_transformer_identifier_small():
    """Fixture providing a small sentence transformer identifier."""
    return "all-minilm-l6-v2"


@pytest.fixture
def openai_identifier():
    """Fixture providing an OpenAI identifier."""
    return "text-embedding-3-small"

@pytest.fixture
def litellm_identifier():
    """Fixture providing an LiteLLM identifier."""
    return "huggingface/microsoft/codebert-base"

@pytest.fixture
def invalid_identifier():
    """Fixture providing an invalid identifier."""
    return "invalid-identifier-123"


def test_auto_embeddings_model2vec(model2vec_identifier):
    """Test that the AutoEmbeddings class can get model2vec embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(model2vec_identifier)
    assert isinstance(embeddings, Model2VecEmbeddings)
    assert embeddings.model_name_or_path == model2vec_identifier


def test_auto_embeddings_sentence_transformer(sentence_transformer_identifier):
    """Test that the AutoEmbeddings class can get sentence transformer embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(sentence_transformer_identifier)
    assert isinstance(embeddings, SentenceTransformerEmbeddings)
    assert embeddings.model_name_or_path == sentence_transformer_identifier


def test_auto_embeddings_sentence_transformer_alt(
    sentence_transformer_identifier_small,
):
    """Test that the AutoEmbeddings class can get a small sentence transformer embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(sentence_transformer_identifier_small)
    assert isinstance(embeddings, SentenceTransformerEmbeddings)
    assert embeddings.model_name_or_path == sentence_transformer_identifier_small


def test_auto_embeddings_openai(openai_identifier):
    """Test that the AutoEmbeddings class can get OpenAI embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(
        openai_identifier, api_key="your_openai_api_key"
    )
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == openai_identifier

def test_auto_embeddings_litellm(litellm_identifier):
    """Test that the AutoEmbeddings class can get LiteLLM embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(
        litellm_identifier, api_key="your_litellm_api_key"
    )
    assert isinstance(embeddings, LiteLLMEmbeddings)
    assert embeddings.model == litellm_identifier


def test_auto_embeddings_invalid_identifier(invalid_identifier):
    """Test that the AutoEmbeddings class raises an error for an invalid identifier."""
    with pytest.raises(ValueError):
        AutoEmbeddings.get_embeddings(invalid_identifier)


if __name__ == "__main__":
    pytest.main()
