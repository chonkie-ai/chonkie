"""A utility module for handling tokenization across different backends."""
import importlib
import inspect
from collections import defaultdict
from typing import Any, Callable, Dict, List, Union


class Tokenizer:
    """Unified tokenizer interface for Chonkie.
    
    Handles tokenizer initialization and operations across different backends
    (HuggingFace, TikToken, custom tokenizers).
    
    Args:
        tokenizer: Tokenizer instance or identifier (e.g., "gpt2")
        
    Raises:
        ImportError: If required tokenizer backend is not installed

    """
    
    def __init__(self, tokenizer: Union[str, Callable, Any] = "gpt2"):
        """Initialize the tokenizer."""
        # Initialize the tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = self._load_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer
        
        # Determine the tokenizer backend
        self._tokenizer_backend = self._get_tokenizer_backend()

    def _get_tokenizer_backend(self) -> str:
        """Determine the tokenizer backend."""
        if "transformers" in str(type(self.tokenizer)):
            return "transformers"
        elif "tokenizers" in str(type(self.tokenizer)):
            return "tokenizers"
        elif "tiktoken" in str(type(self.tokenizer)):
            return "tiktoken"
        elif callable(self.tokenizer) or inspect.isfunction(self.tokenizer) or inspect.ismethod(self.tokenizer):
            return "callable"
        else:
            raise ValueError(f"Tokenizer backend {str(type(self.tokenizer))} not supported")
    
    def _load_tokenizer(self, tokenizer_name: str):
        """Load a tokenizer based on the backend."""
        try:
            if importlib.util.find_spec("tokenizers") is not None:
                from tokenizers import Tokenizer
                return Tokenizer.from_pretrained(tokenizer_name)
            else:
                raise Warning("Tokenizers library not found. Trying tiktoken.")
        except Exception:
            try:
                if importlib.util.find_spec("tiktoken") is not None:
                    from tiktoken import get_encoding
                    return get_encoding(tokenizer_name)
                else:
                    raise Warning("TikToken library not found. Trying transformers.")
            except Exception:
                try:
                    if importlib.util.find_spec("transformers") is not None:
                        from transformers import AutoTokenizer
                        return AutoTokenizer.from_pretrained(tokenizer_name)
                    else:
                        raise ValueError(
                            "Tokenizer not found in the following libraries: transformers, tokenizers, tiktoken"
                        )
                except Exception:
                    raise ValueError(
                        "Tokenizer not found in the following libraries: transformers, tokenizers, tiktoken"
                    )

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.encode(text, add_special_tokens=False)
        elif self._tokenizer_backend == "tokenizers":
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode(text)
        elif self._tokenizer_backend == "callable":
            raise NotImplementedError("Callable tokenizer backend does not support encoding.")
        else:
            raise ValueError(f"Tokenizer backend {self._tokenizer_backend} not supported.")

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)["input_ids"]
        elif self._tokenizer_backend == "tokenizers":
            return [t.ids for t in self.tokenizer.encode_batch(texts, add_special_tokens=False)]
        elif self._tokenizer_backend == "tiktoken":
            return self.tokenizer.encode_batch(texts)
        elif self._tokenizer_backend == "callable":
            raise NotImplementedError("Callable tokenizer backend does not support batch encoding.")
        else:
            raise ValueError(f"Tokenizer backend {self._tokenizer_backend} not supported.")

    def decode(self, tokens: List[int]) -> str:
        """Decode token ids back to text."""
        if self._tokenizer_backend == "callable":
            raise NotImplementedError("Callable tokenizer backend does not support decoding.")
        return self.tokenizer.decode(tokens)

    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode multiple token lists."""
        if self._tokenizer_backend == "transformers":
            return self.tokenizer.batch_decode(token_lists, skip_special_tokens=True)
        elif self._tokenizer_backend in ["tokenizers", "tiktoken"]:
            return self.tokenizer.decode_batch(token_lists)
        elif self._tokenizer_backend == "callable":
            raise NotImplementedError("Callable tokenizer backend does not support batch decoding.")
        else:
            raise ValueError(f"Tokenizer backend {self._tokenizer_backend} not supported.")

    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text."""
        if self._tokenizer_backend == "transformers":
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        elif self._tokenizer_backend == "tokenizers":
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        elif self._tokenizer_backend == "tiktoken":
            return len(self.tokenizer.encode(text))
        elif self._tokenizer_backend == "callable":
            return self.tokenizer(text)
        else:
            raise ValueError(f"Tokenizer backend {self._tokenizer_backend} not supported.")

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        if self._tokenizer_backend == "transformers":
            return [len(token_list) for token_list in self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)["input_ids"]]
        elif self._tokenizer_backend == "tokenizers":
            return [len(token_list) for token_list in [t.ids for t in self.tokenizer.encode_batch(texts, add_special_tokens=False)]]
        elif self._tokenizer_backend == "tiktoken":
            return [len(token_list) for token_list in self.tokenizer.encode_batch(texts)]
        elif self._tokenizer_backend == "callable":
            return [self.tokenizer(text) for text in texts]
        else:
            raise ValueError(f"Tokenizer backend {self._tokenizer_backend} not supported.")
    
class CharacterTokenizer:
    """Character-based tokenizer."""

    def __init__(self):
        """Initialize the tokenizer."""
        # Initialize the vocabulary with a space character
        self.vocab = []
        self.token2id = defaultdict(lambda: len(self.vocab))

        # Add space character to vocabulary
        _ = self.token2id[' ']
        self.vocab.append(' ')
    
    def get_vocab(self) -> List[str]:
        """Get the vocabulary."""
        return self.vocab
    
    def get_token2id(self) -> Dict[str, int]:
        """Get the token to id mapping."""
        return self.token2id
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        ids = []
        for token in text:
            token_id = self.token2id[token]
            if token_id >= len(self.vocab):
                self.vocab.append(token)
            ids.append(token_id)
        return ids
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids back to text."""
        try:
            return ''.join([self.vocab[token] for token in tokens])
        except IndexError:
            raise ValueError(f"Token {tokens} not found in vocabulary.")
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode multiple token lists."""
        return [self.decode(token_list) for token_list in token_lists]
    
    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text."""
        return len(text)
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        return [len(text) for text in texts]

    def __repr__(self) -> str:
        """Return a string representation of the tokenizer."""
        return f"CharacterTokenizer(vocab_size={len(self.vocab)})"
    
class WordTokenizer:
    """Word-based tokenizer."""
    
    def __init__(self):
        """Initialize the tokenizer."""
        # Initialize the vocabulary with a space character
        self.vocab = []
        self.token2id = defaultdict(lambda: len(self.vocab))

        # Add space character to vocabulary
        _ = self.token2id[' ']
        self.vocab.append(' ')
        
    def get_vocab(self) -> List[str]:
        """Get the vocabulary."""
        return self.vocab
    
    def get_token2id(self) -> Dict[str, int]:
        """Get the token to id mapping."""
        return self.token2id
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        words = text.split(' ')
        return words
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            token_id = self.token2id[token]
            if token_id >= len(self.vocab):
                self.vocab.append(token)
            ids.append(token_id)
        return ids
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids back to text."""
        try:
            return ' '.join([self.vocab[token] for token in tokens])
        except IndexError:
            raise ValueError(f"Token {tokens} not found in vocabulary.")
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode multiple token lists."""
        return [self.decode(token_list) for token_list in token_lists]
    
    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text."""
        return len(self.encode(text))
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        return [len(self.encode(text)) for text in texts]
    
    def __repr__(self) -> str:
        """Return a string representation of the tokenizer."""
        return f"WordTokenizer(vocab_size={len(self.vocab)})"