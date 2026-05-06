# 📄 AI-Powered Document Search & Summarization

A semantic search and summarization tool for PDF, DOCX, and TXT documents. Upload your files, ask natural-language questions, and get concise summaries backed by source passages.

Built as a 2-day hackathon project at Developers Institute.

---

## 🎯 What It Does

1. **Ingests** PDF, DOCX, and TXT documents through a web UI or auto-detected file drops
2. **Splits** them into overlapping chunks for fine-grained retrieval
3. **Embeds** each chunk into a 384-dimensional vector using a sentence-transformer
4. **Indexes** the vectors in FAISS for fast similarity search
5. **Answers** natural-language queries by retrieving the most relevant chunks and summarizing them with a transformer model
6. **Shows sources** alongside every summary so users can verify the answer

---

## 🏗️ Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Upload     │   │   Extract    │   │ Clean+Chunk  │   │    Embed     │
│ (PDF/DOCX/   │ : │ (pypdf/docx) │ : │   (regex)    │ : │  (MiniLM)    │
│   TXT)       │   │              │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                 │
                                                                 ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Display    │   │  Summarize   │   │ Search top-k │   │ FAISS Index  │
│  (Streamlit) │ : │ (DistilBART) │ : │   (cosine)   │ : │ (IndexFlatIP)│
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                              ▲
                                              │
                                       ┌──────┴───────┐
                                       │ Embed Query  │
                                       │  (MiniLM)    │
                                       └──────────────┘
```

---

## 📁 Project Structure

```
doc_search_app/
│
├── app.py                  # Streamlit UI (main entry point)
├── ingestion.py            # Extract text from PDF/DOCX/TXT
├── preprocessing.py        # Clean and chunk text
├── embeddings.py           # Generate embeddings + FAISS index management
├── search.py               # Query embedding + similarity search
├── summarizer.py           # Summarize retrieved chunks
├── watcher.py              # Auto-process new uploads (bonus)
├── evaluate.py             # Test set + metrics
├── config.py               # All settings in one place
│
├── data/
│   ├── uploads/            # User-uploaded documents
│   └── index/              # Saved FAISS index + metadata
│
├── requirements.txt
└── README.md
```

---

## 🧩 Components

| Module | Purpose | Key Library |
|---|---|---|
| `ingestion.py` | Extract text from files | pypdf, python-docx |
| `preprocessing.py` | Clean and chunk text | re (regex) |
| `embeddings.py` | Generate vectors, manage index | sentence-transformers, faiss-cpu |
| `search.py` | Semantic similarity search | faiss-cpu |
| `summarizer.py` | Generate concise summaries | transformers (DistilBART) |
| `watcher.py` | Auto-process new uploads | watchdog |
| `app.py` | Streamlit web UI | streamlit |
| `evaluate.py` | Precision@k / Recall@k / ROUGE | rouge-score |
| `config.py` | Centralized settings | pathlib |

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd doc_search_app
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ First-time install downloads ~2 GB (PyTorch + transformers). Be patient.
> The first run of the app will then download ~580 MB of model weights, which are cached locally afterward.

---

## 🚀 Running the App

### Main app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### File watcher (optional)

In a **separate terminal** (with the venv activated):

```bash
python watcher.py
```

Now any PDF, DOCX, or TXT file dropped into `data/uploads/` will be automatically processed and indexed.

### Evaluation

```bash
python evaluate.py
```

Prints Precision@3, Recall@3, and ROUGE scores for the test cases defined inside the script. Edit the `TEST_CASES` list to match your own documents.

---

## 📊 Evaluation Results

The system was tested on 5 foundational ML papers (BERT, DistilBERT, LoRA, RAG, Attention is All You Need) plus a helper text file, with 8 hand-crafted queries and 6 reference summaries.

### Search Quality

| Metric | Score |
|---|---|
| Average Precision@3 | 0.417 |
| **Average Recall@3** | **1.000** |

**Recall@3 of 1.000** is the headline result: across all 8 queries, every relevant document appeared in the top 3 retrieved chunks. The system never failed to find the right source.

The lower Precision@3 is largely a labeling artifact: most queries had only 1 expected document out of 3 retrieved positions, so the theoretical max was 0.33 for those cases. MRR (Mean Reciprocal Rank) would be a fairer metric for this setup; the correct document was almost always at rank 1.

### Summarization Quality

| Metric | Score |
|---|---|
| Average ROUGE-1 F1 | 0.206 |
| Average ROUGE-2 F1 | 0.032 |
| Average ROUGE-L F1 | 0.139 |

ROUGE measures word overlap, not factual correctness. Inspecting the actual outputs revealed that several "weak overlap" summaries were factually correct but used different vocabulary than the reference. For example, on a RAG query, the system correctly summarized that "RAG models are more factual and specific than BART" — true and on-topic, but ROUGE penalized it for not matching the reference's wording.

A more semantic metric like BERTScore would likely show much stronger summary quality. See `REFLECTION.md` for full analysis.

---

## 🧠 Design Decisions

**Why FAISS instead of Pinecone?**
In-process, no network dependency, no API keys, no rate limits. Ideal for hackathon scale (hundreds-thousands of chunks). Pinecone would be the natural upgrade path for multi-user or millions of vectors.

**Why MiniLM-L6-v2 for embeddings?**
384 dimensions, ~80 MB, very fast on CPU, good quality/speed tradeoff. The standard choice for prototyping semantic search. Recall@3 of 1.000 on the test set validates this choice.

**Why DistilBART instead of full BART?**
Roughly 2x faster than `bart-base` with comparable summary quality on news-style text. CPU-friendly.

**Why 300-word chunks with 50-word overlap?**
- 300 words is roughly 400 tokens, which fits MiniLM's 512-token window with margin
- Overlap prevents losing context when an idea spans a chunk boundary

**Why `IndexFlatIP` and not IVF or HNSW?**
Brute-force search is fast enough at our scale and gives exact (not approximate) results. IVF/HNSW are for millions of vectors and require training data — overkill here.

**Why Streamlit instead of Flask?**
Hackathon-friendly: a working UI in 60 lines of pure Python, with file upload, spinners, and reactive updates built in. Flask would have meant writing HTML, CSS, JavaScript, and routes from scratch.

---

## ⚡ Performance Notes

| Operation | Approx. time (CPU) |
|---|---|
| Indexing | ~1-2 sec per page |
| Search | < 100 ms per query |
| Summarization | 5-8 sec per query |

Summarization is the bottleneck on CPU. On GPU it drops below 1 second, but the project was scoped for CPU-only.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit**: web UI
- **sentence-transformers** (MiniLM-L6-v2): embeddings
- **FAISS** (CPU): vector index
- **Hugging Face transformers** (DistilBART): summarization
- **pypdf**, **python-docx**: text extraction
- **watchdog**: filesystem event monitoring
- **rouge-score**: evaluation metrics

---

## ⚠️ Known Limitations

- **No OCR**: scanned (image-only) PDFs will not extract text. A future version could integrate Tesseract.
- **English-optimized**: MiniLM works best on English; multilingual models (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) are a drop-in replacement if needed.
- **PDF layout issues**: standard pypdf struggles with multi-column academic papers. A layout-aware parser like GROBID would help.
- **Single-user**: no authentication or per-user document isolation.
- **Summaries can hallucinate**: the model occasionally invents details not in the source. Always check the source chunks shown in the UI.
- **No streaming output**: queries block until the summary is fully generated.
- **ROUGE is a noisy metric** for this kind of corpus, undervaluing factually correct but differently-worded summaries.

---

## 🔮 Future Work

- Replace ROUGE with BERTScore or LLM-as-judge for fairer summary evaluation
- Add MRR (Mean Reciprocal Rank) for search evaluation
- Implement hybrid search (BM25 + dense vectors) to handle vocabulary-mismatch queries
- Add OCR for scanned PDFs (Tesseract or PaddleOCR)
- Replace `IndexFlatIP` with HNSW once dataset grows beyond ~50k chunks
- Use a layout-aware PDF parser (GROBID or `unstructured`) for academic papers
- Stream summary tokens to the UI for better perceived performance
- Add per-user document collections with simple authentication
- Add a "delete document" feature to remove entries from the index
- Swap DistilBART for a larger model (e.g., `pegasus-xsum`) when running on GPU

---

## 📝 License

MIT — feel free to use, modify, and learn from this code.

---

## 🙏 Acknowledgements

- Sentence-Transformers team for `all-MiniLM-L6-v2`
- Hugging Face for the transformers ecosystem
- Facebook AI for FAISS
- Streamlit team for making Python UIs painless

Built for the Developers Institute hackathon.
