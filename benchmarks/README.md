# 🦛 CHONK vs The World

> Look how **THICC** the competition is! Meanwhile, Chonkie be looking slim and trim 💪

Ever wondered how much CHONKier other text splitting libraries are? Well, wonder no more! We've put Chonkie up against some of the most popular RAG libraries out there, and the results are... well, let's just say Moto Moto might need to revise his famous quote! 

## ⚡ Speed Benchmarks

> ZOOOOOM! Watch Chonkie run! 🏃‍♂️💨

### 100K Wikipedia Articles
The following benchmarks were run on 100,000 Wikipedia articles from the 
[`chonkie-ai/wikipedia-100k`](https://huggingface.co/datasets/chonkie-ai/wikipedia-100k) dataset

All tests were run on a Google Colab A100 instance.

#### Token Chunking

| Library | Time | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 58 sec | 1x (Im fast af boi) |
| 🔗 LangChain | 1 min 10 sec | 1.21x slower |
| 📚 LlamaIndex | 50 min | 51.7x slower |

#### Sentence Chunking

| Library | Time | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 59 sec | 1x (solo CHONK) |
| 📚 LlamaIndex | 3 min 59 sec | 4.05x slower |
| 🔗 LangChain | N/A | Doesn't exist |

#### Recursive Chunking

| Library | Time | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 1 min 19 sec | 1x (🔃🔃) |
| 🔗 LangChain | 2 min 45 sec | 2.09x slower |
| 📚 LlamaIndex | N/A | Doesn't exist |

#### Semantic Chunking

Tested with `sentence-transformers/all-minilm-l6-v2` model unless specified otherwise.

| Library | Time | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie (with default settings) | 13 min 59 sec | 1x (understanding chonk) |
| 🦛 Chonkie | 1 hour 8 min min 53 sec |  4.92x slower |
| 🔗 LangChain | 1 hour 13 sec | 4.35x slower |
| 📚 LlamaIndex | 1 hour 24 min 15 sec| 6.07x slower |

### 500K Wikipedia Articles
The following benchmarks were run on 500,000 Wikipedia articles from the 
[`chonkie-ai/wikipedia-500k`](https://huggingface.co/datasets/chonkie-ai/wikipedia-500k) dataset

All tests were run on a `c3-highmem-4` VM from Google Cloud with 32 GB RAM and a 200 GB SSD Persistent Disk attachment.

#### Token Chunking

| Library | Time | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 2 min 17 sec | 1x (Im fast af boi) |
| 🔗 LangChain | 2 min 42 sec | 1.18x slower |
| 📚 LlamaIndex | 50 min | 21.9x slower |

#### Sentence Chunking

| Library | Time | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 7 min 16 sec | 1x (solo CHONK) |
| 📚 LlamaIndex | 10 min 55 sec | 1.5x slower |
| 🔗 LangChain | N/A | Doesn't exist |

#### Recursive Chunking

| Library | Time | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 3 min 42 sec | 1x (🔃🔃) |
| 🔗 LangChain | 7 min 36 sec | 2.05x slower |
| 📚 LlamaIndex | N/A | Doesn't exist |

### Paul Graham Essays Dataset

The following benchmarks were run on the Paul Graham Essays dataset using the GPT-2 tokenizer. 

#### Token Chunking

| Library | Time (ms) | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 8.18 | 1x (fastest CHONK) |
| 🔗 LangChain | 8.68 | 1.06x slower |
| 📚 LlamaIndex | 272 | 33.25x slower |

#### Sentence Chunking 

| Library | Time (ms) | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 52.6 | 1x (solo CHONK) |
| 📚 LlamaIndex | 91.2 | 1.73x slower |
| 🔗 LangChain | N/A | Doesn't exist |

#### Semantic Chunking 

| Library | Time | Speed Factor |
|---------|------|--------------|
| 🦛 Chonkie | 482ms | 1x (smart CHONK) |
| 🔗 LangChain | 899ms | 1.86x slower |
| 📚 LlamaIndex | 1.2s | 2.49x slower |


## 📊 Size Comparison (Package Size)

### Default Installation (Basic Chunking)

| Library | Size | Chonk Factor |
|---------|------|--------------|
| 🦛 Chonkie | 11.2 MiB | 1x (base CHONK) |
| 🔗 LangChain | 80 MiB | ~7.1x CHONKier |
| 📚 LlamaIndex | 171 MiB | ~15.3x CHONKier |

### With Semantic Features

| Library | Size | Chonk Factor |
|---------|------|--------------|
| 🦛 Chonkie | 585 MiB | 1x (semantic CHONK) |
| 🔗 LangChain | 625 MiB | ~1.07x CHONKier |
| 📚 LlamaIndex | 678 MiB | ~1.16x CHONKier |

## 💡 Why These Numbers Matter

### Speed Benefits
1. **Faster Processing**: Chonkie leads in all chunking methods!
2. **Production Ready**: Optimized for real-world usage
3. **Consistent Performance**: Fast across all chunking types
4. **Scale Friendly**: Process more text in less time

### Size Benefits
1. **Faster Installation**: Less to download = faster to get started
2. **Lower Memory Footprint**: Lighter package = less RAM usage
3. **Cleaner Dependencies**: Only install what you actually need
4. **CI/CD Friendly**: Faster builds and deployments

Remember what Chonkie always says:
> "I may be a hippo, but I'm still light and fast!" 🦛✨

---

*Note: All measurements were taken using Python 3.8+ on a clean virtual environment. Your actual mileage may vary slightly depending on your specific setup and dependencies.*