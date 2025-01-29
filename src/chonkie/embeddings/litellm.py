import importlib
from litellm import embedding
from litellm import token_counter
from typing import Callable, List, Optional
import os
import time
import numpy as np

from .base import BaseEmbeddings


class LiteLLMEmbeddings(BaseEmbeddings):

    def __init__(
        self,
        model: str = 'huggingface/microsoft/codebert-base',
        input: List[str] = "Hello, my dog is cute",
        user: str = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
        api_base: Optional[str] = None,
        encoding_format: Optional[str] = None,
        timeout: Optional[int] = 300,
        input_type: Optional[str] = "feature-extraction",
    ):
        """Initialize LiteLLM embeddings.

        Args:
            model: Name of the LiteLLM embedding model to use
            input: Text to embed
            user: User ID for API requests
            dimensions: Number of dimensions for the embedding model
            api_key: API key for the model
            api_type: Type of API to use
            api_version: Version of the API to use
            api_base: Base URL for the API
            encoding_format: Encoding format for the input text
            timeout: Timeout in seconds for API requests

        """
        super().__init__()
        if not self.is_available():
            raise ImportError(
                "LiteLLM package is not available. Please install it via pip."
            )
        else:
            # Check if LiteLLM works with given parameters
            try:
                api_key = api_key if api_key is not None else os.environ.get("HUGGINGFACE_API_KEY")
                my_list = []
                my_list.append(input)
                response = embedding(model=model, input=my_list, user=user, dimensions=dimensions, api_key=api_key, api_type=api_type, api_version=api_version, api_base=api_base, encoding_format=encoding_format, timeout=timeout)
            except Exception as e:
                raise ValueError(f"LiteLLM failed to initialize with the given parameters: {e}")
            else:
                self.kwargs = {
                    "user": user,
                    "dimensions": dimensions,
                    "api_key": api_key,
                    "api_type": api_type,
                    "api_version": api_version,
                    "api_base": api_base,
                    "encoding_format": encoding_format,
                    "timeout": timeout,
                }
                self.model = model
                if dimensions is None:
                    self._dimension = len(response.data[0]['embedding'])
                else: 
                    self._dimension = dimensions
   
    @property
    def dimension(self) -> int:
        return self._dimension
        
    
    def embed(self, text: str) -> "np.ndarray":
        if isinstance(text, str):
            text = [text]
        retries = 5  # Number of retries
        wait_time = 10  # Wait time between retries
        for i in range(retries):
            try:
                response = embedding(model=self.model, input=text, **self.kwargs)
            except Exception as e:
                print(f"Attempt {i+1}/{retries}: Model is still loading, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                break
        embeddings = response.data[0]['embedding']
        return np.array(embeddings)

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        if isinstance(texts, str):
            texts = [texts]
        retries = 5  # Number of retries
        wait_time = 10  # Wait time between retries
        for i in range(retries):
            try:
                responses = embedding(
                    model=self.model,
                    input=texts,
                    **self.kwargs 
                )
                # Exit the loop if successful
            except Exception as e:
                print(f"Attempt {i+1}/{retries}: Model is still loading, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                break

        # response = embedding(model=self.model_name, input=texts, **self.kwargs)
        np_embeddings = []
        # np_embeddings.append([entry['embedding'] for entry in responses.data])
        np_embeddings.extend(np.array(entry['embedding']) for entry in responses["data"])
        return np_embeddings

    def count_tokens(self, text: str) -> int:
        return token_counter(model=self.model, text=text)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        token_list = []
        for i in texts:
            token_list.append(token_counter(model=self.model, text=i))
        return token_list
    
    def _tokenizer_helper(self, string: str) -> int:
        return token_counter(model=self.model, text=str)

    def get_tokenizer_or_token_counter(self) -> "Callable[[str], int]":
        return self._tokenizer_helper
        

    def similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.divide(
            np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v), dtype=float
        )
        

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("litellm") is not None

    def __repr__(self) -> str:
        return f"LiteLLMEmbeddings(model={self.model})"