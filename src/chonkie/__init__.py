"""Main package for Chonkie."""

from .chef import (
    ABBREVIATIONS,
    BaseChef,
    TextChef,
    UnicodeReplacements,
)
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
    CohereEmbeddings,
    Model2VecEmbeddings,
    OpenAIEmbeddings,
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
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

# This hippo grows with every release 🦛✨~
__version__ = "0.5.0"
__name__ = "chonkie"
__author__ = "🦛 Chonkie AI Team"

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
    "OllamaEmbeddings",
    "AutoEmbeddings",
]

# Add all refinery classes to __all__
__all__ += [
    "BaseRefinery",
    "OverlapRefinery",
]

# Add all chef classes to __all__
__all__ += [
    "BaseChef",
    "TextChef",
    "ABBREVIATIONS",
    "UnicodeReplacements",
]
