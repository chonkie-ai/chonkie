# 🦛 Chonkie DOCS

> ugh, do i _need_ to explain how to use chonkie? man, that's a bummer... to be honest, Chonkie is very easy, with little documentation necessary, but just in case, i'll include some here.

# Table of Contents

- [🦛 Chonkie DOCS](#-chonkie-docs)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Basic installation](#basic-installation)
  - [Dependency Tables](#dependency-tables)
    - [Chunker Availability](#chunker-availability)
    - [Embeddings Availability](#embeddings-availability)
    - [Required Dependencies](#required-dependencies)
- [Quick Start](#quick-start)
- [Design CHONKosophy](#design-chonkosophy)
  - [core chonk principles](#core-chonk-principles)
    - [1. 🎯 small but precise](#1--small-but-precise)
    - [2. 🚀 surprisingly quick](#2--surprisingly-quick)
    - [3. 🪶 tiny but complete](#3--tiny-but-complete)
    - [4. 🧠 the clever chonk](#4--the-clever-chonk)
    - [5. 🌱 growing with purpose](#5--growing-with-purpose)
  - [why chunking is needed? (and may always be needed!)](#why-chunking-is-needed-and-may-always-be-needed)
  - [does speed matter while chunking? (tl;dr: yes!!!)](#does-speed-matter-while-chunking-tldr-yes)
  - [but but but... how? how is chonkie so fast?](#but-but-but-how-how-is-chonkie-so-fast)
- [Chunkers](#chunkers)
  - [TokenChunker](#tokenchunker)
  - [WordChunker](#wordchunker)
  - [SentenceChunker](#sentencechunker)
  - [SemanticChunker](#semanticchunker)
  - [SDPMChunker](#sdpmchunker)
- [Embeddings](#embeddings)
  - [BaseEmbeddings](#baseembeddings)
  - [Model2Vec Embeddings](#model2vec-embeddings)
  - [SentenceTransformerEmbeddings](#sentencetransformerembeddings)
  - [OpenAIEmbeddings](#openaiembeddings)
  - [Using AutoEmbeddings](#using-autoembeddings)
  - [Creating Custom Embeddings](#creating-custom-embeddings)
- [API Reference](#api-reference)
  - [Chunk object](#chunk-object)
  - [SentenceChunk object](#sentencechunk-object)
  - [SemanticChunk object](#semanticchunk-object)
- [FAQ](#faq)
  - [Can I run a Chunker multiple times on different texts? Is Chonkie thread-safe?](#can-i-run-a-chunker-multiple-times-on-different-texts-is-chonkie-thread-safe)

# Installation

## Basic installation

```bash
pip install chonkie
```

## Dependency Tables

As per the details mentioned in the [design](#design-chonkosophy) section, Chonkie is lightweight because it keeps most of the dependencies for each chunker seperate, making it more of an aggregate of multiple repositories and python packages. The optional dependencies feature in python really helps with this. 

### Chunker Availability

The following table shows which chunkers are available with different installation options:

| Chunker | Default | embeddings | 'all' |
|---------|---------|-----------|----------|
| TokenChunker | ✅ | ✅ |  ✅ |
| WordChunker | ✅ | ✅ |  ✅ |
| SentenceChunker | ✅ |  ✅ | ✅ |
| SemanticChunker | ❌ |  ✅ | ✅ |
| SDPMChunker | ❌ | ✅ | ✅ |

Any of the embeddings availability will enable the `SemanticChunker` and `SDPMChunker`. Please check the availability of the embeddings below or you may use the `chonkie[semantic]` install for quick access.  

### Embeddings Availability

The following table shows which embedding providers are available with different installation options:

| Embeddings Provider | Default | 'model2vec' | 'st' | 'openai' | 'semantic'| 'all' |
|--------------------|---------|-----------|----------|-------|--------|---------|
| Model2VecEmbeddings | ❌| ✅ | ❌ | ❌ | ✅ | ✅ |
| SentenceTransformerEmbeddings | ❌ | ❌| ✅ | ❌ | ❌ | ✅|
| OpenAIEmbeddings | ❌ | ❌ | ❌ | ✅ | ❌ | ✅|

### Required Dependencies

| Installation Option | Additional Dependencies |
|--------------------|------------------------|
| Default | autotiktokenizer |
| 'model2vec' | + model2vec, numpy |
| 'st' | + sentence-transformers, numpy |
| 'openai' | + openai, tiktoken |
| 'semantic' | + model2vec, numpy |
| 'all' | all above dependencies |

NOTE: We have seperate `semantic` and `all` installs pre-packaged that might match other installation options breeding redundancy. This redundancy is so we can provide the user the best experience with the freedom to choose their prefered means. The `semantic` and `all` optional installs would continue to change in future versions, so what you might expect to download today may not be the same for tomorrow.

You can install the version you need using:

```bash
# Basic installation (TokenChunker, WordChunker, SentenceChunker)
pip install chonkie

# For the default semantic provider support
pip install "chonkie[semantic]"

# For OpenAI embeddings support
pip install "chonkie[openai]"

# For installing multiple features together 
pip install "chonkie[st, model2vec]" 

# For all features
pip install "chonkie[all]"
```

Note: Installing either 'semantic' or 'openai' extras will enable SemanticChunker and SDPMChunker, as these chunkers can work with any embeddings provider. The difference is in which embedding providers are available for use with these chunkers.

# Quick Start

```python
from chonkie import TokenChunker

# create chunker
chunker = TokenChunker(
    tokenizer="gpt2", # You can pass your desired tokenizer
    chunk_size=512,
    chunk_overlap=128
)

# chunk your text
text = """your long text here..."""
chunks = chunker.chunk(text)

# access chunks
for chunk in chunks:
    print(f"chunk: {chunk.text[:50]}...")
    print(f"tokens: {chunk.token_count}")
```

# Design CHONKosophy

> did you know that pygmy hippos are only 1/4 the size of regular hippos, but they're just as mighty? that's the chonkie spirit - tiny but powerful! 🦛

listen up chonkers! just like our adorable pygmy hippo mascot, chonkie proves that the best things come in small packages. let's dive into why this tiny chonkster is built the way it is!

## core chonk principles

### 1. 🎯 small but precise

like how pygmy hippos take perfect little bites of their favorite fruits, chonkie knows exactly how to size your chunks:

- **compact & efficient**: just like our tiny mascot, every chunk is exactly the size it needs to be
- **smart defaults**: we've done the research so you don't have to! our default parameters are battle-tested
- **flexible sizing**: because sometimes you need a smaller bite!

### 2. 🚀 surprisingly quick

fun fact: pygmy hippos might be small, but they can zoom through the forest at impressive speeds! similarly, chonkie is:

- **lightning fast**: small size doesn't mean slow performance
- **optimized paths**: like our mascot's forest shortcuts, we take the most efficient route (we use cacheing extensively btw!)
- **minimal overhead**: no wasted energy, just pure chonk power

### 3. 🪶 tiny but complete

just as pygmy hippos pack all hippo features into a compact frame, chonkie is:

- **minimum footprint**: base installation smaller than a pygmy hippo footprint
- **modular growth**: add features as you need them, like a growing hippo
- **zero bloat**: every feature has a purpose, just like every trait of our tiny friend
- **smart imports**: load only what you need, when you need it

### 4. 🧠 the clever chonk

why chunking still matters (from a tiny hippo's perspective):

1. **right-sized processing**
   - even tiny chunks can carry big meaning
   - smart chunking = efficient processing
   - our pygmy hippo philosophy: "just enough, never too much"

2. **the goldilocks zone**
   - too small: like a hippo bite that's too tiny
   - too large: like trying to swallow a whole watermelon
   - just right: the chonkie way™️ (pygmy-approved!)

3. **semantic sense**
   - each chunk is carefully crafted
   - like our mascot's careful step through the forest
   - small, meaningful units that work together

### 5. 🌱 growing with purpose

like how pygmy hippos stay small but mighty, chonkie grows sensibly:

```
smart chunks → better embeddings → precise retrieval → quality generation
```

even as models grow bigger, you'll appreciate our tiny-but-mighty approach:

- focused context (like a pygmy hippo's keen senses)
- efficient processing (like our mascot's energy-saving size)
- clean, purposeful design (like nature's perfect mini-hippo)

## why chunking is needed? (and may always be needed!)

while you might be aware of models having longer and longer contexts in recent times (as of 2024), models have yet to reach the stage where adding additional context to them comes for free. additional context, even with the greatest of model architectures comes at a o(n) penalty in speed, to say nothing of the additional memory requirements. and as long as we belive in that attention is all we need, it doesn't seem likely we would be free from this penalty. 

that means, to make models run efficiently (lower speed, memory) it is absoulutely vital that we provide the most accurate information it needs during the retrieval phase. 

accuracy is one part during retrieval and the other is granularity. you might be able to extract the relevant article out for model to work with, but if only 1 line is relevant from that passage, you are in effect adding a lot of noise that would hamper and confuse the model in practice. you want and hope to give the model only what it should require ideally (of course, the ideal scenario is rarely ever possible). this finally brings us to granularity and retrieval accuracy. 

representation models (or embedding models as you may call them) are great at representing large amount of information (sometimes pages of text) in a limited space of just 700-1000 floats, but that doesn't mean it does not suffer from any loss. most representation is lossy, and if we have many concepts being covered in the same space, it is often that much of it would be lost. however, singluar concepts and explainations breed stronger representation vectors. it then becomes vital again to make sure we don't dilute the representation with noise. 

all this brings me back to chunking. chunking, done well, can make sure your representation vector (or embedding) is of high-quality to be able to retrieve the best context for your model to generate with. and that in turn, leads to better quality rag generations. therefore, i believe chunking is here to stay as long as rag is here. and hence, it becomes important that we give it little more than a after-thought. 

## does speed matter while chunking? (tl;dr: yes!!!)

human time is limited, and if you have an option that gives you faster chunks, why would you not?

but speed is not just a bonus; it's central to chonkie! whether you are doing rag on the entirity of wikipedia or working for large scale organization data that updates regularly, you would need the speed that chonkie comes with. stock solutions just don't cut it in these scenarios.

## but but but... how? how is chonkie so fast?

we used a lot of optimizations when building each and every chunker inside chonkie, making sure it's as optimized as possible.

1. **using tiktoken (as a default):** tiktoken is around 3-6x faster than it's counterparts; and it is blazing fast when used with multiple threads. we see the available threads on the cpu at the moment, and use about ~70-80% of them (so as to not hog all resources), which inturn let's us tokenize fast.
2. **pre-compute and cache:** we never tokenize or embed on the fly! as long as something can be pre-computed and cached we do that, store it and re-use it wherever possible. ram is cheap but time is priceless. (of course, we also provide options to turn off the pre-computation and make it memory efficient if need be)
3. **running mean pooling:** most semantic chunkers re-embed the chunks every time they get updated, but we don't do that. we pre-compute the embeddings for the sentences, and use mathematical trickery (which is theoretically found) to instead have a running mean pooling of tokens -- which allows us to save the cost from the embedding models. 

# Chunkers

## TokenChunker

The `TokenChunker` splits text into chunks based on token count.

**Key Parameters:**

- `tokenizer` (`Optional[str, tokenizers.Tokenizer, tiktoken.Encoding]`): any tokenizer implementing the encode/decode interface
- `chunk_size` (`int`): maximum tokens per chunk
- `chunk_overlap` (`Union[int, float]`): number of overlapping tokens between chunks

**Methods:**

- `chunk`: Chunks a piece of text.
  - **Parameters:**
    - `text` (`str`): The input text to be chunked.
  - **Returns:**
    - `List[Chunk]`: A list of `Chunk` objects containing the chunked text and metadata.

- `chunk_batch`: Chunks a list of strings.
  - **Parameters:**
    - `texts` (`List[str]`): A list of input texts to be chunked.
  - **Returns:**
    - `List[List[Chunk]]`: A list of lists of `Chunk` objects, where each sublist corresponds to the chunks of an input text.

- `__call__`: Takes either a string or a list of strings for chunking.
  - **Parameters:**
    - `text` (`Union[str, List[str]]`): The input text or list of texts to be chunked.
  - **Returns:**
    - `Union[List[Chunk], List[List[Chunk]]]`: A list of `Chunk` objects if a single string is provided, or a list of lists of `Chunk` objects if a list of strings is provided.

**Example Usage:**

```python
# Import the TokenChunker
from chonkie import TokenChunker
from autotiktokenizer import AutoTikTokenizer

# Initialize the tokenizer
tokenizer = AutoTikTokenizer.from_pretrained("gpt2")

# Initialize the chunker
chunker = TokenChunker(
    tokenizer=tokenizer,
    chunk_size=512,  # maximum tokens per chunk
    chunk_overlap=128  # overlap between chunks
)

# Chunk a single piece of text
chunks = chunker.chunk("Woah! Chonkie, the chunking library is so cool! I love the tiny hippo hehe.")
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")

# Chunk a batch of texts
texts = ["First text to chunk.", "Second text to chunk."]
batch_chunks = chunker.chunk_batch(texts)
for text_chunks in batch_chunks:
    for chunk in text_chunks:
        print(f"Chunk: {chunk.text}")
        print(f"Tokens: {chunk.token_count}")

# Use the chunker as a callable
chunks = chunker("Another text to chunk using __call__.")
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")
```

## WordChunker

The `WordChunker` maintains word boundaries while chunking, ensuring words stay intact.

**Key Parameters:**

- `tokenizer` (`Optional[str, tokenizers.Tokenizer, tiktoken.Encoding]`): Any tokenizer implementing the encode/decode interface
- `chunk_size` (`int`): Maximum tokens per chunk
- `chunk_overlap` (`int`): Number of overlapping tokens between chunks
- `mode` (`str`): Chunking mode, either 'simple' (space-based splitting) or 'advanced' (handles punctuation and special cases)

**Methods:**

- `chunk`: Chunks a piece of text.
  - **Parameters:**
    - `text` (`str`): The input text to be chunked.
  - **Returns:**
    - `List[Chunk]`: A list of `Chunk` objects containing the chunked text and metadata.

- `chunk_batch`: Chunks a list of strings.
  - **Parameters:**
    - `texts` (`List[str]`): A list of input texts to be chunked.
  - **Returns:**
    - `List[List[Chunk]]`: A list of lists of `Chunk` objects, where each sublist corresponds to the chunks of an input text.

- `__call__`: Takes either a string or a list of strings for chunking.
  - **Parameters:**
    - `text` (`Union[str, List[str]]`): The input text or list of texts to be chunked.
  - **Returns:**
    - `Union[List[Chunk], List[List[Chunk]]]`: A list of `Chunk` objects if a single string is provided, or a list of lists of `Chunk` objects if a list of strings is provided.

**Example Usage:**

```python
from chonkie import WordChunker
from autotiktokenizer import AutoTikTokenizer

tokenizer = AutoTikTokenizer.from_pretrained("gpt2")

chunker = WordChunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128,
    mode="advanced"
)

# Chunk a single piece of text
chunks = chunker.chunk("Some text to chunk while preserving word boundaries.")
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")
```

## SentenceChunker

The `SentenceChunker` preserves sentence boundaries while chunking text.

**Key Parameters:**

- `tokenizer` (`Optional[str, tokenizers.Tokenizer, tiktoken.Encoding]`): Any tokenizer implementing the encode/decode interface
- `chunk_size` (`int`): Maximum tokens per chunk
- `chunk_overlap` (`int`): Number of overlapping tokens between chunks
- `min_sentences_per_chunk` (`int`): Minimum number of sentences to include in each chunk

**Methods:**

- `chunk`: Chunks a piece of text.
  - **Parameters:**
    - `text` (`str`): The input text to be chunked.
  - **Returns:**
    - `List[SentenceChunk]`: A list of `SentenceChunk` objects containing the chunked text and metadata, including individual sentences.

- `chunk_batch`: Chunks a list of strings.
  - **Parameters:**
    - `texts` (`List[str]`): A list of input texts to be chunked.
  - **Returns:**
    - `List[List[SentenceChunk]]`: A list of lists of `SentenceChunk` objects.

- `__call__`: Takes either a string or a list of strings for chunking.
  - **Parameters:**
    - `text` (`Union[str, List[str]]`): The input text or list of texts to be chunked.
  - **Returns:**
    - `Union[List[SentenceChunk], List[List[SentenceChunk]]]`: A list of `SentenceChunk` objects or a list of lists of `SentenceChunk` objects.

**Example Usage:**

```python
from chonkie import SentenceChunker
from autotiktokenizer import AutoTikTokenizer

tokenizer = AutoTikTokenizer.from_pretrained("gpt2")

chunker = SentenceChunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128,
    min_sentences_per_chunk=1
)

# Chunk a single piece of text
chunks = chunker.chunk("First sentence. Second sentence. Third sentence.")
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Number of sentences: {len(chunk.sentences)}")
```

## SemanticChunker

The `SemanticChunker` groups content by semantic similarity. The implementation is inspired by the semantic chunking approach described in the [FullStackRetrieval Tutorials](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb), with modifications and optimizations for better performance and integration with Chonkie's architecture.

This version of `SemanticChunker` has some optimizations that speed it up considerably, but make the assumption that the `tokenizer` you used is the same as the one used for `embedding_model`. This is a valid assumption since most often than not, `chunk_size` and hence, `token_count` is dependent on the `embedding_model` context sizes rather than on the Generative models context length.

**Key Parameters:**

- `embedding_model` (`Union[str, SentenceTransformer]`): Model for semantic embeddings, either a model name string or a SentenceTransformer instance
- `similarity_threshold` (`Optional[float]`): Minimum similarity score to consider sentences similar (0-1)
- `similarity_percentile` (`Optional[float]`): Minimum similarity percentile to consider sentences similar (0-100)
- `chunk_size` (`Optional[int]`): Maximum tokens allowed per chunk
- `initial_sentences` (`Optional[int]`): Number of sentences to start each chunk with

**Methods:**

- `chunk`: Chunks a piece of text using semantic similarity.
  - **Parameters:**
    - `text` (`str`): The input text to be chunked.
  - **Returns:**
    - `List[SemanticChunk]`: A list of `SemanticChunk` objects containing semantically coherent chunks.

- `chunk_batch`: Chunks a list of strings.
  - **Parameters:**
    - `texts` (`List[str]`): A list of input texts to be chunked.
  - **Returns:**
    - `List[List[SemanticChunk]]`: A list of lists of `SemanticChunk` objects.

- `__call__`: Takes either a string or a list of strings for chunking.
  - **Parameters:**
    - `text` (`Union[str, List[str]]`): The input text or list of texts to be chunked.
  - **Returns:**
    - `Union[List[SemanticChunk], List[List[SemanticChunk]]]`: A list of `SemanticChunk` objects or a list of lists of `SemanticChunk` objects.

**Example Usage:**

```python
from chonkie import SemanticChunker

chunker = SemanticChunker(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=512,
    similarity_threshold=0.7
)

# Chunk a single piece of text
chunks = chunker.chunk("Some text with semantic meaning to chunk appropriately.")
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Number of semantic sentences: {len(chunk.sentences)}")
```

## SDPMChunker

the `SDPMChunker` groups content via the semantic double-pass merging method, which groups paragraphs that are semantically similar even if they do not occur consecutively, by making use of a skip-window.

**Key Parameters:**

- `embedding_model` (`Union[str, SentenceTransformer]`): Model for semantic embeddings, either a model name string or a SentenceTransformer instance
- `similarity_threshold` (`Optional[float]`): Minimum similarity score to consider sentences similar (0-1)
- `similarity_percentile` (`Optional[float]`): Minimum similarity percentile to consider sentences similar (0-100)
- `chunk_size` (`Optional[int]`): Maximum tokens allowed per chunk
- `initial_sentences` (`Optional[int]`): Number of sentences to start each chunk with
- `skip_window` (`Optional[int]`): Number of chunks to skip when looking for similarities

**Methods:**

- `chunk`: Chunks a piece of text using semantic double-pass merging.
  - **Parameters:**
    - `text` (`str`): The input text to be chunked.
  - **Returns:**
    - `List[SemanticChunk]`: A list of `SemanticChunk` objects containing semantically coherent chunks.

- `chunk_batch`: Chunks a list of strings.
  - **Parameters:**
    - `texts` (`List[str]`): A list of input texts to be chunked.
  - **Returns:**
    - `List[List[SemanticChunk]]`: A list of lists of `SemanticChunk` objects.

- `__call__`: Takes either a string or a list of strings for chunking.
  - **Parameters:**
    - `text` (`Union[str, List[str]]`): The input text or list of texts to be chunked.
  - **Returns:**
    - `Union[List[SemanticChunk], List[List[SemanticChunk]]]`: A list of `SemanticChunk` objects or a list of lists of `SemanticChunk` objects.

**Example Usage:**

```python
from chonkie import SDPMChunker

chunker = SDPMChunker(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=512,
    similarity_threshold=0.7,
    skip_window=1
)

# Chunk a single piece of text
chunks = chunker.chunk("Some text with related but non-consecutive content to chunk.")
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Number of semantic sentences: {len(chunk.sentences)}")
```

# Embeddings

Chonkie provides a flexible embeddings system that can be used with various embedding providers. The embeddings system is designed to work seamlessly with the semantic chunking features.

## BaseEmbeddings

All embedding implementations in Chonkie inherit from the `BaseEmbeddings` abstract class, which defines the common interface:

```python
class BaseEmbeddings:
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a vector."""
        pass
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts into vectors."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text."""
        pass
    
    def similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute similarity between two embeddings."""
        pass
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass
```

## Model2Vec Embeddings

Uses distilled static embedding models with help of [`model2vec`](https://github.com/MinishLab/model2vec) package. These models are 500x faster than standard `SentenceTransformer` models and about 15x smaller with the `potion-base-8M` being just about 30MB. When used in conjuction with `chonkie[model2vec]` the entire package for `SemanticChunker` usage is just about 57MiB, the smallest of all the options and a 10x smaller package size than the other stock options. 

```python
from chonkie.embeddings import Model2VecEmbeddings, AutoEmbeddings

# Initialise with the Model2VecEmbeddings class
embeddings = Model2VecEmbeddings("minishlab/potion-base-8M")

# OR initialise with the AutoEmbeddings get_embeddings()
embeddings = AutoEmbeddings.get_embeddings("minishlab/potion-base-8M")

chunker = SemanticChunker(
    embedding_model=embeddings,
    similarity_threshold=0.5,
)
```

Available potion models from [Minish lab](https://minishlab.github.io/):

- [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)
- [potion-base-4M](https://huggingface.co/minishlab/potion-base-4M)
- [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M)

Resources:

- Model2Vec [blog](https://minishlab.github.io/hf_blogpost/)

## SentenceTransformerEmbeddings

Uses Sentence Transformers models for creating embeddings.

```python
from chonkie.embeddings import SentenceTransformerEmbeddings

# Initialize with default model
embeddings = SentenceTransformerEmbeddings()

# Use specific model
embeddings = SentenceTransformerEmbeddings("paraphrase-MiniLM-L6-v2")

# Use with semantic chunker
chunker = SemanticChunker(embedding_model=embeddings)
```

## OpenAIEmbeddings

Uses OpenAI's API for creating embeddings.

```python
from chonkie.embeddings import OpenAIEmbeddings

# Initialize with API key
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="your-api-key"  # Optional if OPENAI_API_KEY env var is set
)

# Configure batch size and timeouts
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    batch_size=32,
    timeout=30.0
)

# Use with semantic chunker
chunker = SemanticChunker(embedding_model=embeddings)
```

Available OpenAI models:

- `text-embedding-3-small` (1536 dimensions, best performance/cost ratio)
- `text-embedding-3-large` (3072 dimensions, highest performance)
- `text-embedding-ada-002` (1536 dimensions, legacy model)

## Using AutoEmbeddings

The `AutoEmbeddings` class provides a convenient way to load embeddings:

```python
from chonkie.embeddings import AutoEmbeddings

# Load sentence transformers
embeddings = AutoEmbeddings.get_embeddings("sentence-transformers/all-MiniLM-L6-v2")

# Load OpenAI embeddings
embeddings = AutoEmbeddings.get_embeddings(
    "openai/text-embedding-3-small",
    api_key="your-api-key"
)

# Use directly with semantic chunker
chunker = SemanticChunker(
    embedding_model="openai/text-embedding-3-small",
    api_key="your-api-key"
)
```

## Creating Custom Embeddings

You can create custom embedding implementations by inheriting from `BaseEmbeddings`:

```python
from chonkie.embeddings import BaseEmbeddings

class CustomEmbeddings(BaseEmbeddings):
    def embed(self, text: str) -> np.ndarray:
        # Implement embedding logic
        pass
    
    def count_tokens(self, text: str) -> int:
        # Implement token counting
        pass
    
    # Implement other required methods...

# Register with the embeddings registry
EmbeddingsRegistry.register(
    "custom",
    CustomEmbeddings,
    pattern=r"^custom/|^model-name"
)
```

# API Reference

## Chunk object

```python
@dataclass
class Chunk:
    text: str           # the chunk text
    start_index: int    # starting position in original text
    end_index: int      # ending position in original text
    token_count: int    # number of tokens in chunk
```

## SentenceChunk object

```python
@dataclass
class Sentence: 
    text: str
    start_index: int
    end_index: int
    token_count: int

@dataclass
class SentenceChunk(Chunk):
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentences: list[Sentence] 
```

## SemanticChunk object

```python
@dataclass
class SemanticSentence(Sentence): 
    text: str
    start_index: int
    end_index: int
    token_count: int
    embedding: optional[np.ndarray]

@dataclass
class SemanticChunk(SentenceChunk):
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentences: list[SemanticSentence] 
```

# FAQ

## Can I run a Chunker multiple times on different texts? Is Chonkie thread-safe? 

Yes! Chonkie's Chunkers can be run multiple times without having to re-initialize them. Just initialise them once like you would expect to, and run them on any piece of text you might want to.

That also means it is absolutely thread-safe! But I would recommend monitoring the CPU usage, since few Chunkers frequently default to multi-threaded chunking (like WordChunker and SentenceChunker) so your resources might be depleted faster than usual running these Chunkers.
