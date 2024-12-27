"""Dataclasses for Chonkie."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

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

    def __post_init__(self):
        """Validate the Context attributes after initialization."""
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.token_count, int):
            raise ValueError("token_count must be an integer")
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
        return Chunk(
            text=self.text,
            start_index=self.start_index,
            end_index=self.end_index,
            token_count=self.token_count,
        )


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

@dataclass
class RecursiveLevel:
    """Configuration for a single level of recursive chunking.

    Attributes:
        delimiters: The delimiters to use for the level. If None, that level will use tokens to determine chunk boundaries.
        whitespace: Whether to use whitespace as a delimiter.

    """

    delimiters: Union[List[str], str, None] = None
    whitespace: bool = False

    def __post_init__(self):
        """Post-initialize the recursive level."""
        self.validate()
    
    def validate(self):
        """Validate the recursive level."""
        if self.delimiters is not None and self.whitespace:
            raise ValueError("Cannot have both delimiters and whitespace. "
                             "Use two separate levels instead, one for whitespace and one for delimiters.")
        if self.delimiters is not None:
            for delimiter in self.delimiters:
                if not isinstance(delimiter, str):
                    raise ValueError("All delimiters must be strings")
                if len(delimiter) == 0:
                    raise ValueError("All delimiters must be non-empty strings")
                if delimiter == " ":
                    raise ValueError("Cannot use whitespace as a delimiter",
                                     "Use whitespace=True instead")
                
    def __repr__(self) -> str:
        """Get a string representation of the recursive level."""
        return f"RecursiveLevel(delimiters={self.delimiters}, whitespace={self.whitespace})"
    
    def __str__(self) -> str:
        """Get a string representation of the recursive level."""
        return f"RecursiveLevel(delimiters={self.delimiters}, whitespace={self.whitespace})"

@dataclass
class RecursiveRules: 
    """Collection of rules for recursive chunking."""

    levels: Union[List[RecursiveLevel], RecursiveLevel, None] = None

    def __post_init__(self):
        """Initialize the recursive rules if not already initialized."""
        # Set default levels if not already initialized
        if self.levels is None:
            # First level should be paragraphs
            paragraph_level = RecursiveLevel(delimiters=["\n\n", "\n", "\r\n"],
                                              whitespace=False)
            # Second level should be sentences
            sentence_level = RecursiveLevel(delimiters=[".", "?", "!"],
                                            whitespace=False)
            
            # Third level can be sub-sentences, like '...', ',', ';', ':', etc.
            sub_sentence_level = RecursiveLevel(delimiters=[',',
                                                             ';',
                                                             ':',
                                                             '...',
                                                             '-',
                                                             '(',
                                                             ')',
                                                             '[',
                                                             ']',
                                                             '{',
                                                             '}',
                                                             '<',
                                                             '>',
                                                             '|',
                                                             '~',
                                                             '`',
                                                             '\'',
                                                             '\"'
                                                             ],
                                                 whitespace=False)

            # Fourth level should be words
            word_level = RecursiveLevel(delimiters=None,
                                        whitespace=True)
            # Fifth level should be tokens
            # NOTE: When delimiters is None, the level will use tokens to determine chunk boundaries.
            token_level = RecursiveLevel(delimiters=None,
                                        whitespace=False)
            self.levels = [paragraph_level,
                            sentence_level,
                            sub_sentence_level,
                            word_level,
                            token_level]
        else:
            if isinstance(self.levels, RecursiveLevel):
                self.levels.validate()
            elif isinstance(self.levels, list) and all(isinstance(level, RecursiveLevel) for level in self.levels):
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
    
    def __str__(self) -> str:
        """Get a string representation of the recursive rules."""
        return f"RecursiveRules(levels={self.levels})"


@dataclass
class RecursiveChunk(Chunk):
    """A Chunk with a level attribute."""

    level: Union[int, None] = None

    def __repr__(self) -> str:
        """Get a string representation of the recursive chunk."""
        return (f"RecursiveChunk(text={self.text}, "
                f"start_index={self.start_index}, "
                f"end_index={self.end_index}, "
                f"token_count={self.token_count}, "
                f"level={self.level})")
    
    def __str__(self) -> str:
        """Get a string representation of the recursive chunk."""
        return (f"RecursiveChunk(text={self.text}, "
                f"start_index={self.start_index}, "
                f"end_index={self.end_index}, "
                f"token_count={self.token_count}, "
                f"level={self.level})")
