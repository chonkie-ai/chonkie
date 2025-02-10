"""Main package for Chonkie."""

from .chunker import (
    BaseChunker,
    LateChunker,
    RecursiveChunker,
    SDPMChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
    WordChunker,
)
from .embeddings import (
    AutoEmbeddings,
    BaseEmbeddings,
    Model2VecEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    CohereEmbeddings,
)
from .refinery import (
    BaseRefinery,
    OverlapRefinery,
)
from .tokenizer import CharacterTokenizer, Tokenizer, WordTokenizer
from .types import (
    Chunk,
    Context,
    LateChunk,
    RecursiveChunk,
    RecursiveLevel,
    RecursiveRules,
    SemanticChunk,
    SemanticSentence,
    Sentence,
    SentenceChunk,
)

__version__ = "0.4.1"
__name__ = "chonkie"
__author__ = "Chonkie AI"

# Add basic package metadata to __all__
__all__ = [
    "__name__",
    "__version__",
    "__author__",
]

# Add all data classes to __all__
__all__ += [
    "Context",
    "Chunk",
    "RecursiveChunk",
    "RecursiveLevel",
    "RecursiveRules",
    "SentenceChunk",
    "SemanticChunk",
    "Sentence",
    "SemanticSentence",
    "LateChunk",
]

# Add all tokenizer classes to __all__
__all__ += [
    "Tokenizer",
    "CharacterTokenizer",
    "WordTokenizer",
]

# Add all chunker classes to __all__
__all__ += [
    "BaseChunker",
    "TokenChunker",
    "WordChunker",
    "SentenceChunker",
    "SemanticChunker",
    "SDPMChunker",
    "LateChunker",
    "RecursiveChunker",
]

# Add all embeddings classes to __all__
__all__ += [
    "BaseEmbeddings",
    "Model2VecEmbeddings",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "AutoEmbeddings",
]

# Add all refinery classes to __all__
__all__ += [
    "BaseRefinery",
    "OverlapRefinery",
]