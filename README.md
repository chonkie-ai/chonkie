<div align='center'>

![Chonkie Logo](/assets/chonkie_logo_br_transparent_bg.png)

# 🦛 Chonkie ✨

[![PyPI version](https://img.shields.io/pypi/v/chonkie.svg)](https://pypi.org/project/chonkie/)
[![License](https://img.shields.io/github/license/bhavnicksm/chonkie.svg)](https://github.com/bhavnicksm/chonkie/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-chonkie.ai-blue.svg)](https://docs.chonkie.ai)
![Package size](https://img.shields.io/badge/size-9.7MB-blue)
[![Downloads](https://static.pepy.tech/badge/chonkie)](https://pepy.tech/project/chonkie)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/nMYNVyuB5Y?style=flat)](https://discord.gg/nMYNVyuB5Y)
[![GitHub stars](https://img.shields.io/github/stars/bhavnicksm/chonkie.svg)](https://github.com/bhavnicksm/chonkie/stargazers)

_The no-nonsense RAG chunking library that's lightweight, lightning-fast, and ready to CHONK your texts_

[Installation](#installation) •
[Usage](#usage) •
[Supported Methods](#supported-methods) •
[Benchmarks](#benchmarks-️) •
[Documentation](https://docs.chonkie.ai) •
[Citation](#citation)

</div>

so i found myself making another RAG bot (for the 2342148th time) and meanwhile, explaining to my juniors about why we should use chunking in our RAG bots, only to realise that i would have to write chunking all over again unless i use the bloated software library X or the extremely feature-less library Y. _WHY CAN I NOT HAVE SOMETHING JUST RIGHT, UGH?_

Can't i just install, import and run chunking and not have to worry about dependencies, bloat, speed or other factors?

Well, with chonkie you can! (chonkie boi is a gud boi)

**🚀 Feature-rich**: All the CHONKs you'd ever need </br>
**✨ Easy to use**: Install, Import, CHONK </br>
**⚡ Fast**: CHONK at the speed of light! zooooom </br>
**🌐 Wide support**: Supports all your favorite tokenizer CHONKS </br>
**🪶 Light-weight**: No bloat, just CHONK </br>
**🦛 Cute CHONK mascot**: psst it's a pygmy hippo btw </br>
**❤️ [Moto Moto](#acknowledgements)'s favorite python library** </br>

What're you waiting for, **just CHONK it**!

# Installation

To install chonkie, simply run:

```bash
pip install chonkie
```

Chonkie follows the rule to have minimal default installs, read the [DOCS](https://docs.chonkie.ai) to know the installation for your required chunker, or simply install `all` if you don't want to think about it (not recommended).

```bash
pip install chonkie[all]
```

# Usage

Here's a basic example to get you started:

```python
# First import the chunker you want from Chonkie 
from chonkie import TokenChunker

# Import your favorite tokenizer library
# Also supports AutoTokenizers, TikToken and AutoTikTokenizer
from tokenizers import Tokenizer 
tokenizer = Tokenizer.from_pretrained("gpt2")

# Initialize the chunker
chunker = TokenChunker(tokenizer)

# Chunk some text
chunks = chunker("Woah! Chonkie, the chunking library is so cool! I love the tiny hippo hehe.")

# Access chunks
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")
```

More example usages given inside the [DOCS](https://docs.chonkie.ai)

# Supported Methods

Chonkie provides several chunkers to help you split your text efficiently for RAG applications. Here's a quick overview of the available chunkers:

- **TokenChunker**: Splits text into fixed-size token chunks.
- **WordChunker**: Splits text into chunks based on words.
- **SentenceChunker**: Splits text into chunks based on sentences.
- **RecursiveChunker**: Splits text hierarchically using customizable rules to create semantically meaningful chunks.
- **SemanticChunker**: Splits text into chunks based on semantic similarity.
- **SDPMChunker**: Splits text using a Semantic Double-Pass Merge approach.
- **LateChunker (experimental)**: Embeds text and then splits it to have better chunk embeddings.

More on these methods and the approaches taken inside the [DOCS](https://docs.chonkie.ai)

# Benchmarks 🏃‍♂️

> "I may be smol hippo, but I pack a punch!" 🦛

Here's a quick peek at how Chonkie performs:

**Size**📦

- **Default Install:** 9.7MB (vs 80-171MB for alternatives)
- **With Semantic:** Still lighter than the competition!

**Speed**⚡

- **Token Chunking:** 33x faster than the slowest alternative
- **Sentence Chunking:** Almost 2x faster than competitors
- **Semantic Chunking:** Up to 2.5x faster than others

Check out our detailed [benchmarks](https://docs.chonkie.ai/benchmarks) to see how Chonkie races past the competition! 🏃‍♂️💨

# Acknowledgements

Chonkie would like to CHONK its way through a special thanks to all the users and contributors who have helped make this library what it is today! Your feedback, issue reports, and improvements have helped make Chonkie the CHONKIEST it can be.

And of course, special thanks to [Moto Moto](https://www.youtube.com/watch?v=I0zZC4wtqDQ&t=5s) for endorsing Chonkie with his famous quote:
> "I like them big, I like them chonkie."
>                                         ~ Moto Moto


# Citation

If you use Chonkie in your research, please cite it as follows:

```bibtex
@misc{chonkie2024,
  author = {Minhas, Bhavnick},
  title = {Chonkie: A Fast Feature-full Chunking Library for RAG Bots},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bhavnick/chonkie}},
}
```
