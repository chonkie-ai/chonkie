"""Dataclasses for Chonkie."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Union

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Context:

    """A dataclass representing contextual information for chunk refinement.

    This class stores text and token count information that can be used to add
    context to chunks during the refinement process. It can represent context
    that comes before or after a chunk.

    Attributes:
        text (str): The context text
        token_count (int): Number of tokens in the context text
        start_index (Optional[int]): Starting position of context in original text
        end_index (Optional[int]): Ending position of context in original text

    Example:
        context = Context(
            text="This is some context.",
            token_count=5,
            start_index=0,
            end_index=20
        )

    """

    text: str
    token_count: int
    start_index: Optional[int] = None
    end_index: Optional[int] = None

    # Trivial function but we keep it for consistency with other chunk types.
    def to_dict(self) -> dict:
        """Return the Context as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict):
        """Create a Context object from a dictionary."""
        return cls(**data)

    def __post_init__(self):
        """Validate the Context attributes after initialization."""
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if (
            self.start_index is not None
            and self.end_index is not None
            and self.start_index > self.end_index
        ):
            raise ValueError("start_index must be less than or equal to end_index")

    def __len__(self) -> int:
        """Return the length of the context text."""
        return len(self.text)

    def __str__(self) -> str:
        """Return a string representation of the Context."""
        return self.text

    def __repr__(self) -> str:
        """Return a detailed string representation of the Context."""
        return (
            f"Context(text='{self.text}', token_count={self.token_count}, "
            f"start_index={self.start_index}, end_index={self.end_index})"
        )


@dataclass
class Chunk:

    """Dataclass representing a text chunk with metadata.

    All attributes are read-only via slots for performance reasons.

    Attributes:
        text: The text content of the chunk
        start_index: The starting index of the chunk in the original text
        end_index: The ending index of the chunk in the original text
        token_count: The number of tokens in the chunk
        context: The context of the chunk, useful for refinery classes

    """

    text: str
    start_index: int
    end_index: int
    token_count: int
    context: Optional[Context] = None

    # Trivial function but we keep it for consistency across chunk types.
    def to_dict(self) -> dict:
        """Return the Chunk as a dictionary."""
        result = self.__dict__.copy()
        result["context"] = self.context.to_dict() if self.context is not None else None
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create a Chunk object from a dictionary."""
        context_repr = data.pop("context")
        return cls(
            **data,
            context=Context.from_dict(context_repr)
            if context_repr is not None
            else None,
        )

    def __str__(self) -> str:
        """Return string representation of the chunk."""
        return self.text

    def __len__(self) -> int:
        """Return the length of the chunk."""
        return len(self.text)

    def __repr__(self) -> str:
        """Return string representation of the chunk."""
        if self.context is not None:
            return (
                f"Chunk(text={self.text}, start_index={self.start_index}, "
                f"end_index={self.end_index}, token_count={self.token_count})"
            )
        else:
            return (
                f"Chunk(text={self.text}, start_index={self.start_index}, "
                f"end_index={self.end_index}, token_count={self.token_count}, "
                f"context={self.context})"
            )

    def __iter__(self):
        """Return an iterator over the chunk."""
        return iter(self.text)

    def __getitem__(self, index: int):
        """Return the item at the given index."""
        return self.text[index]

    def copy(self) -> "Chunk":
        """Return a deep copy of the chunk."""
        return Chunk.from_dict(self.to_dict())


@dataclass
class Sentence:

    """Dataclass representing a sentence with metadata.

    All attributes are read-only via slots for performance reasons.

    Attributes:
        text: The text content of the sentence
        start_index: The starting index of the sentence in the original text
        end_index: The ending index of the sentence in the original text
        token_count: The number of tokens in the sentence

    """

    text: str
    start_index: int
    end_index: int
    token_count: int

    # Trivial functions but we keep them for consistency with other chunk types.
    def to_dict(self) -> dict:
        """Return the Chunk as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict):
        """Create a Sentence object from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        """Return a string representation of the Sentence."""
        return (
            f"Sentence(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count})"
        )


@dataclass
class SentenceChunk(Chunk):

    """Dataclass representing a sentence chunk with metadata.

    All attributes are read-only via slots for performance reasons.

    Attributes:
        text: The text content of the chunk
        start_index: The starting index of the chunk in the original text
        end_index: The ending index of the chunk in the original text
        token_count: The number of tokens in the chunk
        sentences: List of Sentence objects in the chunk

    """

    # Don't redeclare inherited fields
    sentences: List[Sentence] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the SentenceChunk as a dictionary."""
        result = super().to_dict()
        result["sentences"] = [sentence.to_dict() for sentence in self.sentences]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SentenceChunk":
        """Create a SentenceChunk object from a dictionary."""
        sentences_dict = data.pop("sentences") if "sentences" in data else None
        sentences = (
            [Sentence.from_dict(sentence) for sentence in sentences_dict]
            if sentences_dict is not None
            else []
        )
        return cls(**data, sentences=sentences)

    def __repr__(self) -> str:
        """Return a string representation of the SentenceChunk."""
        return (
            f"SentenceChunk(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"sentences={self.sentences})"
        )


@dataclass
class SemanticSentence(Sentence):

    """Dataclass representing a semantic sentence with metadata.

    This class is used to represent a sentence with an embedding.

    Attributes:
        text: The text content of the sentence
        start_index: The starting index of the sentence in the original text
        end_index: The ending index of the sentence in the original text
        token_count: The number of tokens in the sentence
        embedding: The sentence embedding

    """

    embedding: Optional["np.ndarray"] = field(default=None)

    def to_dict(self) -> dict:
        """Return the SemanticSentence as a dictionary."""
        result = super().to_dict()
        result["embedding"] = (
            self.embedding.tolist() if self.embedding is not None else None
        )
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create a SemanticSentence object from a dictionary."""
        embedding_list = data.pop("embedding")
        # NOTE: We can't use np.array() here because we don't import numpy in this file,
        # and we don't want add 50MiB to the package size.
        embedding = embedding_list if embedding_list is not None else None
        return cls(**data, embedding=embedding)

    def __repr__(self) -> str:
        """Return a string representation of the SemanticSentence."""
        return (
            f"SemanticSentence(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"embedding={self.embedding})"
        )


@dataclass
class SemanticChunk(SentenceChunk):

    """SemanticChunk dataclass representing a semantic chunk with metadata.

    Attributes:
        text: The text content of the chunk
        start_index: The starting index of the chunk in the original text
        end_index: The ending index of the chunk in the original text
        token_count: The number of tokens in the chunk
        sentences: List of SemanticSentence objects in the chunk

    """

    sentences: List[SemanticSentence] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the SemanticChunk as a dictionary."""
        result = super().to_dict()
        result["sentences"] = [sentence.to_dict() for sentence in self.sentences]
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create a SemanticChunk object from a dictionary."""
        sentences_dict = data.pop("sentences")
        sentences = [
            SemanticSentence.from_dict(sentence) for sentence in sentences_dict
        ]
        return cls(**data, sentences=sentences)

    def __repr__(self) -> str:
        """Return a string representation of the SemanticChunk."""
        return (
            f"SemanticChunk(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"sentences={self.sentences})"
        )


@dataclass
class LateSentence(Sentence):

    """LateSentence dataclass representing a sentence with an embedding.

    This class is used to represent a sentence with an embedding.

    Attributes:
        text: The text content of the sentence
        start_index: The starting index of the sentence in the original text
        end_index: The ending index of the sentence in the original text
        token_count: The number of tokens in the sentence
        embedding: The sentence embedding

    """

    embedding: Optional["np.ndarray"] = field(default=None)

    def to_dict(self) -> dict:
        """Return the LateSentence as a dictionary."""
        result = super().to_dict()
        result["embedding"] = (
            self.embedding.tolist() if self.embedding is not None else None
        )
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create a LateSentence object from a dictionary."""
        embedding_list = data.pop("embedding")
        embedding = (
            np.array(embedding_list, dtype=np.float64)
            if embedding_list is not None
            else None
        )
        return cls(**data, embedding=embedding)

    def __repr__(self) -> str:
        """Return a string representation of the LateSentence."""
        return (
            f"LateSentence(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"embedding={self.embedding})"
        )


@dataclass
class LateChunk(Chunk):

    """LateChunk dataclass representing a chunk with an embedding.

    This class is used to represent a chunk with an embedding.

    Attributes:
        text: The text content of the chunk
        start_index: The starting index of the chunk in the original text
        end_index: The ending index of the chunk in the original text
        token_count: The number of tokens in the chunk
        embedding: The chunk embedding

    """

    sentences: List[LateSentence] = field(default_factory=list)
    embedding: Optional["np.ndarray"] = field(default=None)

    def to_dict(self) -> dict:
        """Return the LateChunk as a dictionary."""
        result = super().to_dict()
        result["sentences"] = [sentence.to_dict() for sentence in self.sentences]
        result["embedding"] = (
            self.embedding.tolist() if self.embedding is not None else None
        )
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create a LateChunk object from a dictionary."""
        sentences_dict = data.pop("sentences")
        sentences = [LateSentence.from_dict(sentence) for sentence in sentences_dict]
        embedding_list = data.pop("embedding")
        embedding = (
            np.array(embedding_list, dtype=np.float64)
            if embedding_list is not None
            else None
        )
        return cls(**data, sentences=sentences, embedding=embedding)

    def __repr__(self) -> str:
        """Return a string representation of the LateChunk."""
        return (
            f"LateChunk(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"sentences={self.sentences}, embedding={self.embedding})"
        )


@dataclass
class RecursiveLevel:

    """Configuration for a single level of recursive chunking.

    Attributes:
        delimiters: The delimiters to use for the level. If None, that level will use tokens to determine chunk boundaries.
        whitespace: Whether to use whitespace as a delimiter.

    """

    delimiters: Union[List[str], str, None] = None
    whitespace: bool = False
    include_delim: Union[Literal["prev", "next", None], None] = "prev"

    def __post_init__(self):
        """Post-initialize the recursive level."""
        self.validate()

    def validate(self):
        """Validate the recursive level."""
        if self.delimiters is not None and self.whitespace:
            raise ValueError(
                "Cannot have both delimiters and whitespace. "
                "Use two separate levels instead, one for whitespace and one for delimiters."
            )
        if self.delimiters is not None:
            for delimiter in self.delimiters:
                if not isinstance(delimiter, str):
                    raise ValueError("All delimiters must be strings")
                if len(delimiter) == 0:
                    raise ValueError("All delimiters must be non-empty strings")
                if delimiter == " ":
                    raise ValueError(
                        "Cannot use whitespace as a delimiter",
                        "Use whitespace=True instead",
                    )

    def to_dict(self) -> dict:
        """Return the RecursiveLevel as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict):
        """Create a RecursiveLevel object from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        """Return a string representation of the RecursiveLevel."""
        return (
            f"RecursiveLevel(delimiters={self.delimiters}, "
            f"whitespace={self.whitespace}, "
            f"include_delim={self.include_delim})"
        )


@dataclass
class RecursiveRules:

    """Collection of rules for recursive chunking."""

    levels: Union[List[RecursiveLevel], RecursiveLevel, None] = None

    def __post_init__(self):
        """Initialize the recursive rules if not already initialized."""
        # Set default levels if not already initialized
        if self.levels is None:
            # First level should be paragraphs
            paragraph_level = RecursiveLevel(
                delimiters=["\n\n", "\n", "\r\n"], whitespace=False
            )
            # Second level should be sentences
            sentence_level = RecursiveLevel(
                delimiters=[".", "?", "!"], whitespace=False
            )

            # Third level can be sub-sentences, like '...', ',', ';', ':', etc.
            sub_sentence_level = RecursiveLevel(
                delimiters=[
                    ",",
                    ";",
                    ":",
                    "...",
                    "-",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                    "<",
                    ">",
                    "|",
                    "~",
                    "`",
                    "'",
                    '"',
                ],
                whitespace=False,
            )

            # Fourth level should be words
            word_level = RecursiveLevel(delimiters=None, whitespace=True)
            # Fifth level should be tokens
            # NOTE: When delimiters is None, the level will use tokens to determine chunk boundaries.
            token_level = RecursiveLevel(delimiters=None, whitespace=False)
            self.levels = [
                paragraph_level,
                sentence_level,
                sub_sentence_level,
                word_level,
                token_level,
            ]
        else:
            if isinstance(self.levels, RecursiveLevel):
                self.levels.validate()
            elif isinstance(self.levels, list) and all(
                isinstance(level, RecursiveLevel) for level in self.levels
            ):
                for level in self.levels:
                    level.validate()

    def __iter__(self):
        """Iterate over the levels."""
        return iter(self.levels)

    def __getitem__(self, index: int) -> RecursiveLevel:
        """Get a level by index."""
        return self.levels[index]

    def __len__(self) -> int:
        """Get the number of levels."""
        return len(self.levels)

    def __repr__(self) -> str:
        """Get a string representation of the recursive rules."""
        return f"RecursiveRules(levels={self.levels})"

    def to_dict(self) -> dict:
        """Return the RecursiveRules as a dictionary."""
        result = dict()
        result["levels"] = None
        if isinstance(self.levels, RecursiveLevel):
            result["levels"] = self.levels.to_dict()
        elif isinstance(self.levels, list):
            result["levels"] = [level.to_dict() for level in self.levels]
        else:
            raise ValueError("Invalid levels type")
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create a RecursiveRules object from a dictionary."""
        levels_repr = data.pop("levels")
        levels = None
        if levels_repr is not None:
            if isinstance(levels_repr, dict):
                levels = RecursiveLevel.from_dict(levels_repr)
            elif isinstance(levels_repr, list):
                levels = [RecursiveLevel.from_dict(level) for level in levels_repr]
        return cls(levels=levels)


@dataclass
class RecursiveChunk(Chunk):

    """A Chunk with a level attribute."""

    level: Union[int, None] = None

    def __repr__(self) -> str:
        """Get a string representation of the recursive chunk."""
        return (
            f"RecursiveChunk(text={self.text}, "
            f"start_index={self.start_index}, "
            f"end_index={self.end_index}, "
            f"token_count={self.token_count}, "
            f"level={self.level})"
        )

    def __str__(self) -> str:
        """Get a string representation of the recursive chunk."""
        return (
            f"RecursiveChunk(text={self.text}, "
            f"start_index={self.start_index}, "
            f"end_index={self.end_index}, "
            f"token_count={self.token_count}, "
            f"level={self.level})"
        )

    def to_dict(self) -> dict:
        """Return the RecursiveChunk as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict):
        """Create a RecursiveChunk object from a dictionary."""
        return cls(**data)
