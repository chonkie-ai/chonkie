"""Sentence chunker."""
from bisect import bisect_left
from itertools import accumulate
from typing import Any, List, Union

from chonkie.types import Chunk, Sentence, SentenceChunk

from .base import BaseChunker


class SentenceChunker(BaseChunker):
    """SentenceChunker splits the sentences in a text based on token limits and sentence boundaries.

    Args:
        tokenizer: The tokenizer instance to use for encoding/decoding
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
        min_chunk_size: Minimum number of tokens per sentence (defaults to 2)
        approximate: Whether to use approximate token counting (defaults to True)

    Raises:
        ValueError: If parameters are invalid

    """

    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        approximate: bool = True,
        delim: Union[str, List[str]] = [".", "!", "?", "\n"],
        **kwargs
    ):
        """Initialize the SentenceChunker with configuration parameters.

        SentenceChunker splits the sentences in a text based on token limits and sentence boundaries.

        Args:
            tokenizer: The tokenizer instance to use for encoding/decoding
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
            min_chunk_size: Minimum number of tokens per sentence (defaults to 2)
            min_characters_per_sentence: Minimum number of characters per sentence
            approximate: Whether to use approximate token counting (defaults to True)
            delim: Delimiters to split sentences on
        Raises:
            ValueError: If parameters are invalid

        """
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if min_sentences_per_chunk < 1:
            raise ValueError("min_sentences_per_chunk must be at least 1")
        if min_characters_per_sentence < 1:
            raise ValueError("min_characters_per_sentence must be at least 1")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.approximate = approximate
        self.delim = delim
        self.sep = "🦛"

    # TODO: This is a older method of sentence splitting that uses Regex
    # but since Regex in python via re is super slooooow we use a different method
    # that is faster and more accurate. We can keep this method for reference
    # and comparison. And also, we'll need to have a seperate preprocessing
    # to handle the special cases that this method handles.

    # def _split_sentences(self, text: str) -> List[str]:
    #     """Split text into sentences using enhanced regex patterns.

    #     Handles various cases including:
    #     - Standard sentence endings across multiple writing systems
    #     - Quotations and parentheses
    #     - Common abbreviations
    #     - Decimal numbers
    #     - Ellipsis
    #     - Lists and enumerations
    #     - Special punctuation
    #     - Common honorifics and titles

    #     Args:
    #         text: Input text to be split into sentences

    #     Returns:
    #         List of sentences
    #     """
    #     # Define sentence ending punctuation marks from various writing systems
    #     sent_endings = (
    #         r'[!.?։؟۔܀܁܂߹।॥၊။።፧፨᙮᜵᜶᠃᠉᥄᥅᪨᪩᪪᪫᭚᭛᭞᭟᰻᰼᱾᱿'
    #         r'‼‽⁇⁈⁉⸮⸼꓿꘎꘏꛳꛷꡶꡷꣎꣏꤯꧈꧉꩝꩞꩟꫰꫱꯫﹒﹖﹗！．？𐩖𐩗'
    #         r'𑁇𑁈𑂾𑂿𑃀𑃁𑅁𑅂𑅃𑇅𑇆𑇍𑇞𑇟𑈸𑈹𑈻𑈼𑊩𑑋𑑌𑗂𑗃𑗉𑗊𑗋𑗌𑗍𑗎𑗏𑗐𑗑𑗒'
    #         r'𑗓𑗔𑗕𑗖𑗗𑙁𑙂𑜼𑜽𑜾𑩂𑩃𑪛𑪜𑱁𑱂𖩮𖩯𖫵𖬷𖬸𖭄𛲟𝪈｡。]'
    #     )

    #     # Common abbreviations and titles that don't end sentences
    #     abbrevs = (
    #         r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|viz|al|Gen|Col|Fig|Lt|Mt|St"
    #         r"|etc|approx|appt|apt|dept|est|min|max|misc|no|ps|seq|temp|etal"
    #         r"|e\.g|i\.e|vol|vs|cm|mm|km|kg|lb|ft|pd|hr|sec|min|sq|fx|Feb|Mar"
    #         r"|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    #     )

    #     # First, protect periods in known abbreviations
    #     text = re.sub(rf"({abbrevs})\.", r"\1@POINT@", text, flags=re.IGNORECASE)

    #     # Protect decimal numbers
    #     text = re.sub(r"(\d+)\.(\d+)", r"\1@POINT@\2", text)

    #     # Protect ellipsis
    #     text = re.sub(r"\.\.\.", "@ELLIPSIS@", text)

    #     # Protect email addresses and websites
    #     text = re.sub(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", r"@EMAIL@\1@EMAIL@", text)
    #     text = re.sub(r"(https?://[^\s]+)", r"@URL@\1@URL@", text)

    #     # Handle parentheses and brackets
    #     text = re.sub(r'\([^)]*\.[^)]*\)', lambda m: m.group().replace('.', '@POINT@'), text)
    #     text = re.sub(r'\[[^\]]*\.[^\]]*\]', lambda m: m.group().replace('.', '@POINT@'), text)

    #     # Handle quotations with sentence endings
    #     text = re.sub(rf'({sent_endings})"(\s+[A-Z])', r'\1"\n\2', text)

    #     # Handle standard sentence endings
    #     text = re.sub(rf'({sent_endings})(\s+[A-Z"]|\s*$)', r'\1\n\2', text)

    #     # Handle lists and enumerations
    #     text = re.sub(r'(\d+\.)(\s+[A-Z])', r'\1\n\2', text)
    #     text = re.sub(r'([a-zA-Z]\.)(\s+[A-Z])', r'\1\n\2', text)

    #     # Restore protected periods and symbols
    #     text = text.replace("@POINT@", ".")
    #     text = text.replace("@ELLIPSIS@", "...")
    #     text = re.sub(r'@EMAIL@([^@]+)@EMAIL@', r'\1', text)
    #     text = re.sub(r'@URL@([^@]+)@URL@', r'\1', text)

    #     # Split into sentences
    #     sentences = [s.strip() for s in text.split('\n') if s.strip()]

    #     return sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Fast sentence splitting while maintaining accuracy.

        This method is faster than using regex for sentence splitting and is more accurate than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences
            
        Returns:
            List of sentences

        """
        t = text
        for c in self.delim:
            t = t.replace(c, c + self.sep)

        # Initial split
        splits = [s for s in t.split(self.sep) if s != ""]
        # print(splits)

        # Combine short splits with previous sentence
        sentences = []
        current = ""

        for s in splits:
            if len(s.strip()) < self.min_characters_per_sentence:
                current += s
            else:
                if current:
                    sentences.append(current)
                current = s

        if current:
            sentences.append(current)

        return sentences

    def _get_token_counts(self, sentences: List[str]) -> List[int]:
        """Get token counts for a list of sentences in batch.

        Args:
            sentences: List of sentences

        Returns:
            List of token counts for each sentence

        """
        # Batch encode all sentences at once
        encoded_sentences = self._encode_batch(sentences)
        return [len(encoded) for encoded in encoded_sentences]

    def _estimate_token_counts(self, sentences: List[str]) -> int:
        """Estimate token count using character length."""
        CHARS_PER_TOKEN = 6.0  # Avg. char per token for llama3 is b/w 6-7
        if type(sentences) is str:
            return max(1, len(sentences) // CHARS_PER_TOKEN)
        elif type(sentences) is list and type(sentences[0]) is str:
            return [max(1, len(t) // CHARS_PER_TOKEN) for t in sentences]
        else:
            raise ValueError(
                f"Unknown type passed to _estimate_token_count: {type(sentences)}"
            )

    def _get_feedback(self, estimate: int, actual: int) -> float:
        """Validate against the actual token counts and correct the estimates."""
        estimate, actual = max(1, estimate), max(1, actual)
        feedback = max(0.01, 1 - ((estimate - actual) / estimate))
        return feedback

    def _prepare_sentences(self, text: str) -> List[Sentence]:
        """Split text into sentences and calculate token counts for each sentence.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of Sentence objects

        """
        # Split text into sentences
        sentence_texts = self._split_sentences(text)
        if not sentence_texts:
            return []

        # Calculate positions once
        positions = []
        current_pos = 0
        for sent in sentence_texts:
            positions.append(current_pos)
            current_pos += len(sent)  # No +1 space because sentences are already separated by spaces

        if not self.approximate:
            # Get accurate token counts in batch
            token_counts = self._get_token_counts(sentence_texts)
        else:
            # Estimate token counts using character length
            token_counts = self._estimate_token_counts(sentence_texts)

        # Create sentence objects
        return [
            Sentence(
                text=sent, start_index=pos, end_index=pos + len(sent), token_count=count
            )
            for sent, pos, count in zip(sentence_texts, positions, token_counts)
        ]

    # def _prepare_sentences(self, text: str) -> List[Sentence]:
    #     """Prepare sentences with either estimated or accurate token counts."""
    #     # Split text into sentences
    #     sentence_texts = self._split_sentences(text)
    #     if not sentence_texts:
    #         return []

    #     # Calculate positions once
    #     positions = []
    #     current_pos = 0
    #     for sent in sentence_texts:
    #         positions.append(current_pos)
    #         current_pos += len(sent) + 1  # +1 for space/separator

    #     if not self.approximate:
    #         # Get accurate token counts in batch
    #         token_counts = self._get_token_counts(sentence_texts)
    #     else:
    #         # Estimate token counts using character length
    #         token_counts = self._estimate_token_counts(sentence_texts)

    #     # Create sentence objects
    #     return [
    #         Sentence(
    #             text=sent, start_index=pos, end_index=pos + len(sent), token_count=count
    #         )
    #         for sent, pos, count in zip(sentence_texts, positions, token_counts)
    #     ]

    def _create_chunk(self, sentences: List[Sentence], token_count: int) -> Chunk:
        """Create a chunk from a list of sentences.

        Args:
            sentences: List of sentences to create chunk from
            token_count: Total token count for the chunk

        Returns:
            Chunk object

        """
        chunk_text = "".join([sentence.text for sentence in sentences])
        return SentenceChunk(
            text=chunk_text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
            sentences=sentences,
        )

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into overlapping chunks based on sentences while respecting token limits.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        # Get prepared sentences with token counts
        sentences = self._prepare_sentences(text)  # 28mus
        if not sentences:
            return []

        # Pre-calculate cumulative token counts for bisect
        # Add 1 token for spaces between sentences
        token_sums = list(
            accumulate(
                [s.token_count for s in sentences], lambda a, b: a + b, initial=0
            )
        )

        chunks = []
        feedback = 1.0
        pos = 0

        while pos < len(sentences):
            # use updated feedback on the token sums
            token_sums = [int(s * feedback) for s in token_sums]

            # Use bisect_left to find initial split point
            target_tokens = token_sums[pos] + self.chunk_size
            split_idx = bisect_left(token_sums, target_tokens) - 1
            split_idx = min(split_idx, len(sentences))

            # Ensure we include at least one sentence beyond pos
            split_idx = max(split_idx, pos + 1)

            # Handle minimum sentences requirement
            if split_idx - pos < self.min_sentences_per_chunk:
                split_idx = pos + self.min_sentences_per_chunk

            # Get the estimated token count
            estimate = token_sums[split_idx] - token_sums[pos]

            # Get candidate sentences and verify actual token count
            chunk_sentences = sentences[pos:split_idx]
            chunk_text = "".join(s.text for s in chunk_sentences)
            actual = len(self._encode(chunk_text))

            # Given the actual token_count and the estimate, get a feedback value for the next loop
            feedback = self._get_feedback(estimate, actual)
            # print(f"Estimate: {estimate} Actual: {actual} feedback: {feedback}")

            # Back off one sentence at a time if we exceeded chunk size
            while (
                actual > self.chunk_size
                and len(chunk_sentences) > self.min_sentences_per_chunk
            ):
                split_idx -= 1
                chunk_sentences = sentences[pos:split_idx]
                chunk_text = "".join(s.text for s in chunk_sentences)
                actual = len(self._encode(chunk_text))

            chunks.append(self._create_chunk(chunk_sentences, actual))

            # Calculate next position with overlap
            if self.chunk_overlap > 0 and split_idx < len(sentences):
                # Calculate how many sentences we need for overlap
                overlap_tokens = 0
                overlap_idx = split_idx - 1

                while overlap_idx > pos and overlap_tokens < self.chunk_overlap:
                    sent = sentences[overlap_idx]
                    next_tokens = overlap_tokens + sent.token_count + 1  # +1 for space
                    if next_tokens > self.chunk_overlap:
                        break
                    overlap_tokens = next_tokens
                    overlap_idx -= 1

                # Move position to after the overlap
                pos = overlap_idx + 1
            else:
                pos = split_idx

        return chunks

    def __repr__(self) -> str:
        """Return a string representation of the SentenceChunker."""
        return (
            f"SentenceChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk})",
            f"min_characters_per_sentence={self.min_characters_per_sentence}",
            f"approximate={self.approximate}",
            f"delim={self.delim}",
        )
