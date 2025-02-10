"""Base classes for chunking text."""

import importlib
import inspect
import warnings
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List, Union

from tqdm import tqdm

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
        else:
            self.tokenizer = tokenizer_or_token_counter
            self._tokenizer_backend = self._get_tokenizer_backend()
            self.token_counter = self._get_tokenizer_counter()
        
        # Set whether to use multiprocessing or not
        self._use_multiprocessing = True

    def _get_tokenizer_backend(self):
        """Return the backend tokenizer object."""
        if "transformers" in str(type(self.tokenizer)):
            return "transformers"
        elif "tokenizers" in str(type(self.tokenizer)):
            return "tokenizers"
        elif "tiktoken" in str(type(self.tokenizer)):
            return "tiktoken"
        elif (
            callable(self.tokenizer)
            or inspect.isfunction(self.tokenizer)
            or inspect.ismethod(self.tokenizer)
        ):
            return "callable"
        else:
            raise ValueError(
                f"Tokenizer backend {str(type(self.tokenizer))} not supported"
            )

    def _load_tokenizer(self, tokenizer_name: str):
        """Load a tokenizer based on the backend."""
        try:
            if importlib.util.find_spec("tokenizers") is not None:
                from tokenizers import Tokenizer

                self._tokenizer_backend = "tokenizers"
                return Tokenizer.from_pretrained(tokenizer_name)
            else:
                raise Warning("Tokenizers library not found. Trying tiktoken.")
        except Exception:
            try:
                if importlib.util.find_spec("tiktoken") is not None:
                    from tiktoken import get_encoding

                    self._tokenizer_backend = "tiktoken"
                    return get_encoding(tokenizer_name)
                else:
                    raise Warning(
                        "TikToken library not found. Trying transformers."
                    )
            except Exception:
                try:
                    if importlib.util.find_spec("transformers") is not None:
                        from transformers import AutoTokenizer

                        self._tokenizer_backend = "transformers"
                        return AutoTokenizer.from_pretrained(tokenizer_name)
                    else:
                        raise ValueError(
                            "Tokenizer not found in the following libraries: transformers, tokenizers, tiktoken",
                            "Please check your installations, or use a different tokenizer.",
                        )
                except Exception:
                    raise ValueError(
                        "Tokenizer not found in the following libraries: transformers, tokenizers, tiktoken",
                        "Please check your installations, or use a different tokenizer.",
                    )

    def _get_tokenizer_counter(self) -> Callable[[str], int]:
        """Get token counter based on tokenizer backend."""
        if self._tokenizer_backend == "transformers":
            return self._transformers_token_counter
        elif self._tokenizer_backend == "tokenizers":
            return self._tokenizers_token_counter
        elif self._tokenizer_backend == "tiktoken":
            return self._tiktoken_token_counter
        elif self._tokenizer_backend == "callable":
            return self.tokenizer
        else:
            raise ValueError("Tokenizer backend not supported for token counting")

    def _transformers_token_counter(self, text: str) -> int:
        """Token counter for transformers backend."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _tokenizers_token_counter(self, text: str) -> int:
        """Token counter for tokenizers backend."""
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def _tiktoken_token_counter(self, text: str) -> int:
        """Token counter for tiktoken backend."""
        return len(self.tokenizer.encode(text))

    def _encode(self, text: str) -> List[int]:
        """Encode text using the backend tokenizer."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.encode(text, add_special_tokens=False)
        elif self._tokenizer_backend == "tokenizers":
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode(text)
        elif self._tokenizer_backend == "callable":
            raise NotImplementedError(
                "Callable tokenizer backend does not support encoding."
            )
        else:
            raise ValueError(
                f"Tokenizer backend {self._tokenizer_backend} not supported."
            )

    def _encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts using the backend tokenizer."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)["input_ids"]
        elif self._tokenizer_backend == "tokenizers":
            return [
                t.ids
                for t in self.tokenizer.encode_batch(texts, add_special_tokens=False)
            ]
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode_batch(texts)
        elif self._tokenizer_backend == "callable":
            raise NotImplementedError(
                "Callable tokenizer backend does not support batch encoding."
            )
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
        elif self._tokenizer_backend == "callable":
            raise NotImplementedError(
                "Callable tokenizer backend does not support decoding."
            )
        else:
            raise ValueError(
                f"Tokenizer backend {self._tokenizer_backend} not supported."
            )

    def _decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode a batch of token lists using the backend tokenizer."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.batch_decode(token_lists, skip_special_tokens=True)
        elif self._tokenizer_backend == "tokenizers":
            return self.tokenizer.decode_batch(token_lists)
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.decode_batch(token_lists)
        elif self._tokenizer_backend == "callable":
            raise NotImplementedError(
                "Callable tokenizer backend does not support batch decoding."
            )
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
    
    def _process_batch_sequential(self,
                                  texts: List[str],
                                  show_progress_bar: bool = True) -> List[List[Chunk]]:
        """Process a batch of texts sequentially."""
        return [
                self.chunk(t) for t in tqdm(
                    texts,
                    desc="🦛",
                    disable=not show_progress_bar,
                        unit="doc",
                    bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱", 
                    ascii=' o')
        ]
    
    def _process_batch_multiprocessing(self,
                                     texts: List[str],
                                     show_progress_bar: bool = True) -> List[List[Chunk]]:
        """Process a batch of texts using multiprocessing."""
        num_workers = self._determine_optimal_workers()
        total = len(texts)
        chunksize = max(1, min(total // (num_workers * 16), 10)) # Optimize chunk size
        
        with Pool(processes=num_workers) as pool:
            results = []
            with tqdm(total=total,
                     desc="🦛",
                     disable=not show_progress_bar,
                     unit="doc",
                     bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                     ascii=' o') as pbar:
                for result in pool.imap(self.chunk, texts, chunksize=chunksize):
                    results.append(result)
                    pbar.update()
            return results
        
    def chunk_batch(
        self,
        texts: List[str],
        show_progress_bar: bool = True,
    ) -> List[List[Chunk]]:
        """Split a List of texts into their respective chunks.

        By default, this method uses multiprocessing to parallelize the chunking process.

        Args:
            texts: List of input texts to be chunked.
            show_progress_bar: Whether to show a progress bar.
        
        Returns:
            List of lists of Chunk objects containing the chunked text and metadata

        """
        if self._use_multiprocessing:
            return self._process_batch_multiprocessing(texts, show_progress_bar)
        else:
            return self._process_batch_sequential(texts, show_progress_bar)

    def __call__(
        self, text: Union[str, List[str]], show_progress_bar: bool = True
    ) -> Union[List[Chunk], List[List[Chunk]]]:
        """Make the chunker callable directly.

        Args:
            text: Input text or list of texts to be chunked
            show_progress_bar: Whether to show a progress bar (for batch chunking)

        Returns:
            List of Chunk objects or list of lists of Chunk

        """
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, list):
            return self.chunk_batch(text, show_progress_bar)
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def __repr__(self) -> str:
        """Return string representation of the chunker."""
        return f"{self.__class__.__name__}()"
