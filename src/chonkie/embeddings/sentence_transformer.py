"""SentenceTransformer embeddings."""

import importlib.util as importutil
from typing import TYPE_CHECKING, List, Union

from chonkie.embeddings.base import BaseEmbeddings

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer

if importutil.find_spec("sentence_transformers"):
    import numpy as np
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(BaseEmbeddings):

    """Class for SentenceTransformer embeddings.

    This class provides an interface for the SentenceTransformer library, which
    provides a variety of pre-trained models for sentence embeddings. This is also
    the recommended way to use sentence-transformers in Chonkie.

    Args:
        model (str): Name of the SentenceTransformer model to load

    """

    def __init__(
        self, model: Union[str, "SentenceTransformer"] = "all-MiniLM-L6-v2", **kwargs
    ) -> None:
        """Initialize SentenceTransformerEmbeddings with a sentence-transformers model.

        Args:
            model (str): Name of the SentenceTransformer model to load
            **kwargs: Additional keyword arguments to pass to the SentenceTransformer constructor

        Raises:
            ImportError: If sentence-transformers is not available
            ValueError: If the model is not a string or SentenceTransformer instance

        """
        super().__init__()

        if not self.is_available():
            raise ImportError(
                "SentenceTransformer is not available. Please install it via pip."
            )
        else:
            global SentenceTransformer, np
            import numpy as np
            from sentence_transformers import SentenceTransformer

        if isinstance(model, str):
            self.model_name_or_path = model
            self.model = SentenceTransformer(self.model_name_or_path, **kwargs)
        elif isinstance(model, SentenceTransformer):
            self.model = model
            self.model_name_or_path = self.model.model_card_data.base_model
        else:
            raise ValueError("model must be a string or SentenceTransformer instance")

        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> "np.ndarray":
        """Embed a single text using the sentence-transformers model."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Embed multiple texts using the sentence-transformers model."""
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_as_tokens(self, text: str) -> "np.ndarray":
        """Embed the text as tokens using the sentence-transformers model.

        This method is useful for getting the token embeddings of a text. It
        would work even if the text is longer than the maximum sequence length.
        """
        if text == "":
            return np.array([])

        # Use the model's tokenizer to encode the text
        encodings = self.model.tokenizer(text, add_special_tokens=False)["input_ids"]

        max_seq_length = self.max_seq_length
        token_splits = []
        for i in range(0, len(encodings), max_seq_length):
            if i + max_seq_length <= len(encodings):
                token_splits.append(encodings[i : i + max_seq_length])
            else:
                token_splits.append(encodings[i:])

        split_texts = self.model.tokenizer.batch_decode(token_splits)
        # Get the token embeddings
        token_embeddings = self.model.encode(
            split_texts, output_value="token_embeddings", add_special_tokens=False
        )

        # Since SentenceTransformer doesn't automatically convert embeddings into
        # numpy arrays, we need to do it manually
        if (
            type(token_embeddings) is not list
            and type(token_embeddings) is not np.ndarray
        ):
            token_embeddings = token_embeddings.cpu().numpy()
        elif type(token_embeddings) is list and type(token_embeddings[0]) not in [
            np.ndarray,
            list,
        ]:
            token_embeddings = [
                embedding.cpu().numpy() for embedding in token_embeddings
            ]

        # Concatenate the token embeddings
        token_embeddings = np.concatenate(token_embeddings, axis=0)

        # Assertion always fails because of special tokens added by encode process
        # assert token_embeddings.shape[0] == len(encodings), \
        #     (f"The number of token embeddings should be equal to the number of tokens in the text"
        #      f"Expected: {len(encodings)}, Got: {token_embeddings.shape[0]}")

        return token_embeddings

    def embed_as_tokens_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Embed multiple texts as tokens using the sentence-transformers model."""
        return [self.embed_as_tokens(text) for text in texts]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.model.tokenizer.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts using the model's tokenizer."""
        encodings = self.model.tokenizer(texts)
        return [len(enc) for enc in encodings["input_ids"]]

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> float:
        """Compute cosine similarity between two embeddings."""
        return self.model.similarity(u, v).item()

    def get_tokenizer_or_token_counter(self):
        """Return the tokenizer or token counter object."""
        return self.model.tokenizer

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def max_seq_length(self) -> int:
        """Return the maximum sequence length."""
        return self.model.get_max_seq_length()

    @classmethod
    def is_available(cls) -> bool:
        """Check if sentence-transformers is available."""
        return importutil.find_spec("sentence_transformers") is not None

    def __repr__(self):
        """Representation of the SentenceTransformerEmbeddings instance."""
        return f"SentenceTransformerEmbeddings(model={self.model_name_or_path})"
