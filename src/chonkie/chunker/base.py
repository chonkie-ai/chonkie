"""Base classes for chunking text."""

import importlib
import inspect
import warnings
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List, Union

from chonkie.types import Chunk


class BaseChunker(ABC):
    """Abstract base class for all chunker implementations.

    All chunker implementations should inherit from this class and implement
    the chunk() method according to their specific chunking strategy.
    """

    def __init__(
        self, tokenizer_or_token_counter: Union[str, Any, Callable[[str], int]]
    ):
        """Initialize the chunker with a tokenizer.

        Args:
            tokenizer_or_token_counter (Union[str, Any]): String, tokenizer object, or token counter object

        """
        # First check if the tokenizer_or_token_counter is a string
        if isinstance(tokenizer_or_token_counter, str):
            self.tokenizer = self._load_tokenizer(tokenizer_or_token_counter)
            self.token_counter = self._get_tokenizer_counter()
        # Then check if the tokenizer_or_token_counter is a function via inspect or method (e.g., a custom tokenizer function)
        elif inspect.isfunction(tokenizer_or_token_counter) or inspect.ismethod(
            tokenizer_or_token_counter
        ):
            self.tokenizer = None
            self._tokenizer_backend = "callable"
            self.token_counter = tokenizer_or_token_counter
        # If not function or string, then assume it's a tokenizer object
        else:
            self.tokenizer = tokenizer_or_token_counter
            self._tokenizer_backend = self._get_tokenizer_backend()
            self.token_counter = self._get_tokenizer_counter()

    def _get_tokenizer_backend(self):
        """Return the backend tokenizer object."""
        if "transformers" in str(type(self.tokenizer)):
            return "transformers"
        elif "tokenizers" in str(type(self.tokenizer)):
            return "tokenizers"
        elif "tiktoken" in str(type(self.tokenizer)):
            return "tiktoken"
        else:
            raise ValueError(
                f"Tokenizer backend {str(type(self.tokenizer))} not supported"
            )

    def _load_tokenizer(self, tokenizer_name: str):
        """Load a tokenizer based on the backend."""
        try:
            if importlib.util.find_spec("tiktoken") is not None:
                from tiktoken import get_encoding

                self._tokenizer_backend = "tiktoken"
                return get_encoding(tokenizer_name)
            else:
                raise Warning("TikToken library not found. Trying autotiktokenizer.")
        except Exception:
            try:
                if importlib.util.find_spec("autotiktokenizer") is not None:
                    from autotiktokenizer import AutoTikTokenizer

                    self._tokenizer_backend = "tiktoken"
                    return AutoTikTokenizer.from_pretrained(tokenizer_name)
                else:
                    raise Warning(
                        "AutoTikTokenizer library not found. Trying tokenizers."
                    )
            except Exception:
                try:
                    if importlib.util.find_spec("tokenizers") is not None:
                        from tokenizers import Tokenizer

                        self._tokenizer_backend = "tokenizers"
                        return Tokenizer.from_pretrained(tokenizer_name)
                    else:
                        raise Warning(
                            "Tokenizers library not found. Trying transformers."
                        )
                except Exception:
                    try:
                        if importlib.util.find_spec("transformers") is not None:
                            from transformers import AutoTokenizer

                            self._tokenizer_backend = "transformers"
                            return AutoTokenizer.from_pretrained(tokenizer_name)
                    except Exception:
                        raise ValueError(
                            "Tokenizer not found in the following libraries: transformers, tokenizers, autotiktokenizer, tiktoken",
                            "Please install one of these libraries to use the chunker.",
                        )

    def _get_tokenizer_counter(self) -> Callable[[str], int]:
        """Get token counter based on tokenizer backend."""
        if self._tokenizer_backend == "transformers":
            return lambda x: len(self.tokenizer.encode(x, add_special_tokens=False))
        elif self._tokenizer_backend == "tokenizers":
            return lambda x: len(self.tokenizer.encode(x, add_special_tokens=False).ids)
        elif self._tokenizer_backend == "tiktoken":
            return lambda x: len(self.tokenizer.encode(x))
        else:
            raise ValueError("Tokenizer backend not supported for token counting")

    def _encode(self, text: str) -> List[int]:
        """Encode text using the backend tokenizer."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.encode(text, add_special_tokens=False)
        elif self._tokenizer_backend == "tokenizers":
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode(text)
        else:
            raise ValueError(
                f"Tokenizer backend {self._tokenizer_backend} not supported."
            )

    def _encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts using the backend tokenizer."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)[
                "input_ids"
            ]
        elif self._tokenizer_backend == "tokenizers":
            return [
                t.ids
                for t in self.tokenizer.encode_batch(texts, add_special_tokens=False)
            ]
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode_batch(texts)
        else:
            raise ValueError(
                f"Tokenizer backend {self._tokenizer_backend} not supported."
            )

    def _decode(self, tokens) -> str:
        """Decode tokens using the backend tokenizer."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.decode(tokens)
        elif self._tokenizer_backend == "tokenizers":
            return self.tokenizer.decode(tokens)
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.decode(tokens)
        else:
            raise ValueError(
                f"Tokenizer backend {self._tokenizer_backend} not supported."
            )

    def _decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode a batch of token lists using the backend tokenizer."""
        if self._tokenizer_backend == "transformers":
            return [self.tokenizer.decode(tokens) for tokens in token_lists]
        elif self._tokenizer_backend == "tokenizers":
            return [self.tokenizer.decode(tokens) for tokens in token_lists]
        elif self._tokenizer_backend == "tiktoken":
            return [self.tokenizer.decode(tokens) for tokens in token_lists]
        else:
            raise ValueError(
                f"Tokenizer backend {self._tokenizer_backend} not supported."
            )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the backend tokenizer."""
        return self.token_counter(text)

    def _count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts using the backend tokenizer."""
        return [self.token_counter(text) for text in texts]

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks according to the implementation strategy.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        pass

    def _determine_optimal_workers(self) -> int:
        """Determine optimal number of workers based on system resources."""
        try:
            # Get CPU cores
            cpu_cores = cpu_count()

            # Never use more than 75% of available cores
            max_workers = max(1, int(cpu_cores * 0.75))

            # Cap at 8 workers
            return min(max_workers, 8)

        except Exception as e:
            warnings.warn(
                f"Error determining optimal workers: {e}. Using single process."
            )
            return 1

    def chunk_batch(self, text: List[str]) -> List[List[Chunk]]:
        """Split a List of texts into their respective chunks.

        By default, this method uses multiprocessing to parallelize the chunking process.

        Args:
            text: List of input texts to be chunked.

        Returns:
            List of lists of Chunk objects containing the chunked text and metadata

        """
        workers = self._determine_optimal_workers()
        if workers > 1:
            with Pool(workers) as pool:
                return pool.map(self.chunk, text)
        else:
            return [self.chunk(t) for t in text]

    def __call__(
        self, text: Union[str, List[str]]
    ) -> Union[List[Chunk], List[List[Chunk]]]:
        """Make the chunker callable directly.

        Args:
            text: Input text or list of texts to be chunked

        Returns:
            List of Chunk objects or list of lists of Chunk

        """
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, list):
            return self.chunk_batch(text)
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def __repr__(self) -> str:
        """Return string representation of the chunker."""
        return f"{self.__class__.__name__}()"
