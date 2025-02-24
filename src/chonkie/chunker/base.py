"""Base classes for chunking text."""

import warnings
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List, Union

from tqdm import tqdm

from chonkie.tokenizer import Tokenizer
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
        self.tokenizer = Tokenizer(tokenizer_or_token_counter)

        # Set whether to use multiprocessing or not
        self._use_multiprocessing = True

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

    def _process_batch_sequential(
        self, texts: List[str], show_progress_bar: bool = True
    ) -> List[List[Chunk]]:
        """Process a batch of texts sequentially."""
        return [
            self.chunk(t)
            for t in tqdm(
                texts,
                desc="🦛",
                disable=not show_progress_bar,
                unit="doc",
                bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                ascii=" o",
            )
        ]

    def _process_batch_multiprocessing(
        self, texts: List[str], show_progress_bar: bool = True
    ) -> List[List[Chunk]]:
        """Process a batch of texts using multiprocessing."""
        num_workers = self._determine_optimal_workers()
        total = len(texts)
        chunksize = max(1, min(total // (num_workers * 16), 10))  # Optimize chunk size

        with Pool(processes=num_workers) as pool:
            results = []
            with tqdm(
                total=total,
                desc="🦛",
                disable=not show_progress_bar,
                unit="doc",
                bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                ascii=" o",
            ) as pbar:
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
