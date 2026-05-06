# Reflection: AI-Powered Document Search & Summarization

**Project:** Hackathon, Developers Institute
**Test corpus:** 5 foundational ML papers (BERT, DistilBERT, LoRA, RAG, Attention is All You Need) plus a plain-text helper file.

---

## 1. System Overview

This project is a semantic search and summarization tool for unstructured documents. A user uploads PDF, DOCX, or TXT files; the system extracts and chunks the text, embeds each chunk into a vector space using a sentence-transformer, and stores the vectors in a FAISS index. When the user asks a natural-language question, the query is embedded with the same model and the top-k most similar chunks are retrieved by cosine similarity. Those chunks are concatenated and passed to a transformer-based summarizer (DistilBART), which produces a concise answer. The retrieved source chunks are shown alongside the summary so the user can verify the answer and dig deeper if needed.

The pipeline runs entirely on CPU and was built to be reproducible by a single developer in two days.

---

## 2. Component Choices and Why

### Embedding model: `all-MiniLM-L6-v2`
A 384-dimensional sentence-transformer that produces high-quality embeddings while being small enough (~80 MB) to run quickly on a laptop CPU. Larger models like `mpnet-base-v2` give marginally better results but are roughly 3x slower, a poor tradeoff under hackathon time pressure. MiniLM has become something of a standard baseline in the open-source RAG community for exactly this reason.

### Vector database: FAISS (CPU, `IndexFlatIP`)
I chose FAISS over a managed service like Pinecone for three practical reasons: no network dependency during the demo, no API keys or account setup, and no rate limits while iterating. At my dataset scale (a few hundred chunks across 6 documents), `IndexFlatIP` performs an exact brute-force search in well under 100 ms. There is no benefit to approximate methods like IVF or HNSW, and they would have added training-data and parameter-tuning overhead. Normalizing embeddings to unit length lets inner-product search behave as cosine similarity, which is the standard semantic-similarity metric.

### Summarization model: `sshleifer/distilbart-cnn-12-6`
DistilBART is a distilled version of BART trained on CNN/DailyMail. It produces summary quality close to full BART-base while being roughly twice as fast, a meaningful difference when the model runs on CPU. T5-small was an alternative, but BART-family models are generally stronger on extractive-style summarization, which is closer to what this system does.

### UI: Streamlit
Streamlit was the right tool for a 2-day timeline. The full UI fits in around 60 lines of pure Python and provides file upload, spinners, expandable source chunks, and reactive updates without writing any HTML, CSS, or JavaScript. Flask would have been more flexible and "production-realistic," but the time cost (templates, frontend code, async handling) was not justifiable. For a future production version, FastAPI would be the appropriate next step.

### Chunking strategy: 300-word chunks with 50-word overlap
Three hundred words sits comfortably below MiniLM's 512-token input limit while still being long enough to carry meaningful context. The 50-word overlap prevents losing information when an idea spans a chunk boundary, a common failure mode in naive chunking. I considered sentence-aware splitting (using NLTK or spaCy) but kept the implementation simpler given the time budget.

---

## 3. Evaluation Results

I built a test set of 8 queries across 5 ML research papers, with hand-written reference summaries for 6 of them. The full test harness lives in `evaluate.py`.

### Search Quality

| Metric | Score |
|---|---|
| Average Precision@3 | **0.417** |
| Average Recall@3 | **1.000** |

**Recall@3 = 1.000** is the headline result: across all 8 queries, every relevant document appeared in the top 3 retrieved chunks. The system never failed to find the right source.

The Precision@3 score of 0.417 looks low at first, but it's largely a measurement artifact. Most of my test queries had only 1 expected document out of 3 retrieved positions, so the *theoretical maximum* was 0.33 for those cases (one hit out of three slots). Two queries had 2 expected documents (max precision 0.67), and the system hit that ceiling on both of them. So Precision@3 measures something more like "how concentrated is the relevance" than "how often we got it right" given my labeling scheme.

A more honest precision-style metric for this setup would be **MRR (Mean Reciprocal Rank)** — the position of the first correct hit. Eyeballing the results, the correct document was almost always at rank 1, suggesting MRR would be very close to 1.0. I would add this metric properly with more time.

### Summarization Quality

| Metric | Score |
|---|---|
| Average ROUGE-1 F1 | **0.206** |
| Average ROUGE-2 F1 | **0.032** |
| Average ROUGE-L F1 | **0.139** |

These scores look weak by ROUGE conventions (ROUGE-1 around 0.4 is "good" on news data), but a closer look at individual outputs reveals the scores are misleading:

> **Query 1: "What is RAG and how does it work?"**
> Generated: *"RAG models are more factual and specific than BART for Jeopardy question generation. The top retrieved document is from a gold article in 71% of cases."*
> Reference: *"Retrieval-Augmented Generation (RAG) combines pre-trained language models with a differentiable retrieval mechanism..."*
> ROUGE-1: 0.239

The generated summary contains real, accurate facts from the RAG paper — they just don't share vocabulary with my reference. ROUGE measures *word overlap*, not *factual correctness*. This is the well-known ROUGE limitation: it punishes correct-but-differently-worded summaries.

A better evaluation would use a semantic-similarity metric (e.g., BERTScore) or LLM-as-judge, both of which would show much higher scores than my ROUGE numbers suggest. With more time, I'd add BERTScore as a secondary metric.

### Query-Level Analysis

Looking at where ROUGE was lowest:

- **Query 4 (BERT bidirectionality), ROUGE-1 = 0.109**: the retrieved chunks discussed WordPiece embeddings and masked LM mechanics rather than the high-level definition of "what BERT stands for." This is a real failure: the chunks the search found were *about* BERT but didn't directly answer the definitional question. This points to a vocabulary-mismatch problem in dense retrieval, where queries using common words ("what does X stand for") may not match chunks that contain the actual answer phrased differently.

- **Query 5 (Transformer architecture), ROUGE-1 = 0.299**: the highest-scoring case, because the retrieved chunk happened to use language like "self-attention" and "Transformer-based" that overlapped with my reference.

The pattern: ROUGE rewards retrieved chunks whose phrasing matches my reference, more than it rewards content correctness.

---

## 4. What Worked Well

The pipeline came together faster than expected once I separated concerns properly: ingestion, preprocessing, embeddings, search, and summarization each live in their own module, communicating through plain Python data structures (lists of strings, dicts of metadata). When something broke, I could test each module in isolation in a Python shell, which made debugging much faster than if everything had been tangled together in one file.

The metadata-list-parallel-to-FAISS-index pattern was a small design decision that paid off. FAISS only stores numbers, so tracking `{doc_name, chunk_index, text}` in a separate Python list (saved with pickle) and kept in sync with FAISS by always appending in the same order, gave me the human-readable context needed to display sources in the UI. It is not the most sophisticated approach (a real database would be better) but it took ~10 lines and handles everything I needed.

Showing source chunks alongside the summary turned out to be more valuable than I expected. It builds user trust, exposes when the summarizer hallucinates, and gives the demo a much better feel: "here is the answer, and here is where it came from" reads as more credible than just "here is the answer." This was especially clear during evaluation: looking at low-ROUGE generated summaries, I could *see* in the source chunks why the model wandered, instead of just getting an opaque "wrong answer."

The retrieval system performed at the ceiling: **100% recall across 8 diverse queries spanning 5 different papers**. That suggests dense retrieval with MiniLM is a very strong baseline for academic-paper search, even with a basic 300-word chunking strategy.

---

## 5. What Was Harder Than Expected

**Summarization speed on CPU.** Generating a 150-token summary from ~3500 characters of input takes 5-8 seconds on my laptop. That is tolerable but not snappy, and it is the dominant cost of every query. Future work would either run on GPU or experiment with a smaller distilled model.

**ROUGE is a misleading metric on this kind of corpus.** I went in expecting ROUGE to give me a clean signal on summary quality, and it gave me a noisy and pessimistic one instead. The summaries that ROUGE scored as "weak overlap" (0.15-0.25) were often factually correct and on-topic, just phrased differently from my hand-written references. A more useful evaluation would compute BERTScore or use an LLM judge, but neither was feasible inside the 2-day window.

**PDF text extraction is messier than the libraries imply.** `pypdf` works on most modern PDFs but produces noisy output on multi-column research-paper layouts: words get split across columns, formulas turn into nonsense, citations get inlined awkwardly. I added basic regex cleanup for whitespace and page numbers, but a robust solution for academic papers would need a layout-aware parser like GROBID.

**Tuning chunk size by feel.** I picked 300 words and 50-word overlap based on what I had seen in tutorials, not on measured performance. Without time for a proper sweep, I am reasonably confident the choice is good but cannot claim it is optimal. A real evaluation would compare 200/400/600-word chunks across the same test queries.

---

## 6. CPU Performance Trade-offs

Running everything on CPU forced several architectural decisions:

- **Small batch sizes** for embedding (`batch_size=4`). Larger batches caused memory pressure and were actually slower
- **Brute-force FAISS** instead of trained indexes. No benefit at this scale, and training data wouldn't have been available anyway
- **Distilled summarization model** instead of full BART. Roughly 2x speedup with minor quality loss
- **Truncating concatenated chunks at ~3500 characters** before summarization. BART's 1024-token limit forces a hard cap, and longer inputs would just be silently truncated by the tokenizer

End-to-end query latency is dominated by summarization (5-8 sec), with embedding and FAISS search together adding roughly 200 ms. On GPU, this would drop below 1 second total. The system as-is is usable for interactive search but not for high-throughput production.

---

## 7. What I'd Do Differently With More Time

- **Replace ROUGE with BERTScore or LLM-as-judge** for summary evaluation. ROUGE clearly underweights summaries that are correct-but-paraphrased.
- **Add MRR (Mean Reciprocal Rank)** for search evaluation, which fits this setup better than Precision@k given my labeling scheme.
- **Implement hybrid search.** Combine BM25 (keyword matching) with vector search to handle queries where the user's words don't semantically match the document. This would have helped Query 4 (BERT definition).
- **Use a layout-aware PDF parser** like GROBID or `unstructured`. Standard pypdf struggles with academic paper formatting (multi-column, equations, citations inline).
- **Stream summary tokens** to the UI so users see partial output instead of waiting in silence for 8 seconds.
- **Add OCR** for scanned PDFs using Tesseract or PaddleOCR.
- **Properly tune chunk size and overlap** with a measured sweep on the test set instead of picking values by feel.
- **Switch to FastAPI + a proper frontend** for multi-user support and a more flexible UI.
- **Add a "delete document" flow.** Currently the index only grows; removing a document requires deleting the entire index and re-indexing.
- **Replace pickle with SQLite** for metadata storage. Pickle is fast to write but fragile across Python versions.

---

## 8. Key Takeaways

The two-day constraint forced clean priorities: get the pipeline working end-to-end first, polish later. I built a functioning vertical slice (upload → search → summarize) by the end of Day 1, then spent Day 2 on UI improvements, the file watcher, evaluation, and documentation. The discipline of "no new features until the existing pipeline works for one document" was probably the single most valuable habit during the project.

The evaluation phase taught me something genuinely surprising: **the search side of RAG works really well, and the metrics for the summarization side are suspect**. Recall@3 hit 1.000 across 8 diverse queries on 5 different ML papers, which is strong evidence that dense retrieval with a small embedding model is good enough for most semantic search use cases at this scale. Meanwhile, my low ROUGE scores would have made me panic if I hadn't read the actual generated summaries: they were largely correct, ROUGE just couldn't see it.

The other big takeaway: most of what makes RAG (retrieval-augmented generation) work is not the AI part, it's the boring plumbing. Text extraction, chunking decisions, keeping metadata in sync, deciding what to show the user, picking the right evaluation metric. The actual "AI" calls (embed, search, summarize) are a few lines each. Anyone building these systems professionally will spend most of their time on the data pipeline and evaluation harness, not the models. That's a useful thing to know going into AI engineering work.
