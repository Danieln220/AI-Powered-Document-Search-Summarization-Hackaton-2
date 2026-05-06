"""
Evaluation script for the document search & summarization system.
 
Run with:
    python evaluate.py
 
What this measures:
    1. Search quality (Precision@k, Recall@k): did we retrieve
       the correct source document(s)?
    2. Summary quality (ROUGE-1, ROUGE-2, ROUGE-L): does the generated
       summary overlap with a hand-written reference summary?
 
ROUGE explained:
    - ROUGE-1: how many single words (unigrams) the generated summary
      shares with the reference. Measures basic content overlap.
    - ROUGE-2: how many word pairs (bigrams) overlap. Captures whether
      phrases match, not just isolated words.
    - ROUGE-L: longest common subsequence between the two summaries.
      Loosely captures word ORDER, not just word presence.
    - All scores are F1 (balance of precision and recall) and range 0-1.
      For comparison: ROUGE-1 around 0.4 is typical for extractive
      summarization on news data.
 
How to use this for YOUR documents:
    1. Index 3-5 of your school documents through the Streamlit app
       (or by dropping them into data/uploads/ with the watcher running).
    2. Replace the doc names in TEST_CASES below with YOUR actual filenames.
    3. Replace the queries with questions you'd realistically ask about
       those documents.
    4. Write reference_summary strings for at least 3-4 cases (needed for
       ROUGE evaluation). Leave as None for the rest.
    5. Install rouge-score:  pip install rouge-score
    6. Run `python evaluate.py` and paste the results into your reflection.
"""
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from search import search
from summarizer import summarize_chunks
 
 
# ============================================================
# TEST CASES
# ============================================================
# Realistic placeholders for school documents. Replace the
# `expected_docs` filenames with YOUR document filenames once you've
# decided what to index.
#
# Tips for writing good test cases:
#   - Mix easy queries (specific terms from the doc) with harder ones
#     (paraphrased or abstract questions). Gives you something to talk
#     about in your reflection.
#   - Aim for 5-8 cases. Fewer than 5 isn't enough to draw conclusions;
#     more than 8 takes too long to run during evaluation.
#   - For multi-doc queries, put all relevant filenames in expected_docs.
#   - reference_summary is what YOU consider a good answer. 2-3 sentences
#     is plenty. Aim to provide it for at least 3-4 cases so ROUGE
#     averages are meaningful.
 
TEST_CASES = [
    {
        # Easy: specific term directly stated in the RAG abstract
        "query": "What is Retrieval-Augmented Generation and how does it work?",
        "expected_docs": ["RAG_test.pdf"],
        "reference_summary": (
            "Retrieval-Augmented Generation (RAG) combines pre-trained language models "
            "with a differentiable retrieval mechanism over non-parametric memory, "
            "allowing the model to access external documents to answer knowledge-intensive questions."
        ),
    },
    {
        # Easy: LoRA core idea is explicit in the abstract
        "query": "How does LoRA reduce the number of trainable parameters in large language models?",
        "expected_docs": ["LORA_test.pdf"],
        "reference_summary": (
            "LoRA freezes the pre-trained model weights and injects trainable low-rank "
            "decomposition matrices into each Transformer layer, greatly reducing the number "
            "of trainable parameters needed for fine-tuning downstream tasks."
        ),
    },
    {
        # Easy: compression numbers are explicitly in DistilBERT abstract
        "query": "How much smaller and faster is DistilBERT compared to BERT?",
        "expected_docs": ["DISTILLBERT_test.pdf"],
        "reference_summary": (
            "DistilBERT reduces the size of a BERT model by 40% using knowledge distillation "
            "during pre-training, while retaining 97% of its language understanding capabilities "
            "and being 60% faster at inference."
        ),
    },
    {
        # Easy: BERT definition is in the title and abstract
        "query": "What does BERT stand for and what makes it bidirectional?",
        "expected_docs": ["BERT_test.pdf"],
        "reference_summary": (
            "BERT stands for Bidirectional Encoder Representations from Transformers. "
            "It pre-trains deep bidirectional representations by jointly conditioning on both "
            "left and right context in all layers, unlike previous models that read text "
            "left-to-right only."
        ),
    },
    {
        # Easy: Transformer architecture is the entire point of the paper
        "query": "What is the Transformer architecture and why does it replace recurrence?",
        "expected_docs": ["Attention_is_all_you_need.pdf"],
        "reference_summary": (
            "The Transformer is a network architecture based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely. This enables greater "
            "parallelization during training and achieves state-of-the-art results on "
            "machine translation tasks."
        ),
    },
    {
        # Medium: multi-doc — both LoRA and DistilBERT address model efficiency
        "query": "What techniques are used to make large language models more efficient to deploy?",
        "expected_docs": ["LORA_test.pdf", "DISTILLBERT_test.pdf"],
        "reference_summary": (
            "LoRA reduces trainable parameters by injecting low-rank matrices into Transformer "
            "layers, while DistilBERT uses knowledge distillation to compress BERT by 40%, "
            "both making large models significantly cheaper and faster to deploy."
        ),
    },
    {
        # Hard: paraphrased — tests vocabulary mismatch (no word 'external' or 'retrieval' in query)
        "query": "How do neural models access knowledge they were not trained on?",
        "expected_docs": ["RAG_test.pdf"],
        "reference_summary": None,
    },
    {
        # Hard: abstract multi-doc query spanning BERT and Attention paper
        "query": "What role does the attention mechanism play in language model pre-training?",
        "expected_docs": ["BERT_test.pdf", "Attention_is_all_you_need.pdf"],
        "reference_summary": None,
    },
]
 
 
# ============================================================
# SEARCH METRICS
# ============================================================
 
def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@k = (relevant docs in top k) / k
 
    Answers: "Of the top k results we returned, what fraction are
    actually relevant?"
    """
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k if k > 0 else 0.0
 
 
def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Recall@k = (relevant docs in top k) / (total relevant docs)
 
    Answers: "Of all the truly relevant docs, what fraction did we
    find in the top k?"
    """
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant) if relevant else 0.0
 
 
def deduplicate_preserving_order(items: list[str]) -> list[str]:
    """
    Dedupe a list while keeping first-seen order.
 
    Why we need this: a single document gets split into many chunks,
    so search results may have ['doc1.pdf', 'doc1.pdf', 'doc2.pdf'].
    For doc-level precision/recall, we want unique docs.
    """
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
 
 
# ============================================================
# ROUGE SETUP
# ============================================================
# Try to import rouge-score. If it's not installed, we still run
# search metrics but skip summary metrics with a clear message.
#
# Install with: pip install rouge-score
 
try:
    from rouge_score import rouge_scorer
    _scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,  # reduces 'running' and 'run' to the same root
    )
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
 
 
def compute_rouge(generated: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
 
    F1 is the harmonic mean of precision (how much of the generated
    summary appears in the reference) and recall (how much of the
    reference appears in the generated summary). It's the standard
    single-number summary of ROUGE.
    """
    if not ROUGE_AVAILABLE:
        return {}
    scores = _scorer.score(reference, generated)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }
 
 
def interpret_rouge(rouge1: float) -> str:
    """
    Rough interpretation of ROUGE-1 F1 score.
    These thresholds are loose guidelines, not hard rules.
    """
    if rouge1 >= 0.5:
        return "strong overlap"
    elif rouge1 >= 0.3:
        return "decent overlap"
    elif rouge1 >= 0.15:
        return "weak overlap"
    else:
        return "very weak overlap"
 
 
# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
 
def evaluate():
    """Run all test cases and print a results table."""
    print("=" * 72)
    print("  EVALUATION RESULTS")
    print("=" * 72)
 
    if not ROUGE_AVAILABLE:
        print()
        print("  WARNING: rouge-score is NOT installed.")
        print("  Search metrics will run, but ROUGE metrics will be skipped.")
        print("  Install with:  pip install rouge-score")
        print("=" * 72)
 
    precisions, recalls = [], []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
 
    for i, case in enumerate(TEST_CASES, 1):
        # ============ Search step ============
        results = search(case["query"], top_k=5)
        retrieved_docs = [r["doc_name"] for r in results]
        unique_retrieved = deduplicate_preserving_order(retrieved_docs)
 
        p = precision_at_k(unique_retrieved, case["expected_docs"], k=3)
        r = recall_at_k(unique_retrieved, case["expected_docs"], k=3)
        precisions.append(p)
        recalls.append(r)
 
        print(f"\nQuery {i}: {case['query']}")
        print(f"  Retrieved (unique docs): {unique_retrieved[:3]}")
        print(f"  Expected:                {case['expected_docs']}")
        print(f"  Precision@3: {p:.2f}   Recall@3: {r:.2f}")
 
        # ============ Summary step ============
        # Only runs if a reference is provided AND we have search results
        if case.get("reference_summary") and results:
            top_texts = [r["text"] for r in results]
            generated = summarize_chunks(top_texts)
 
            print(f"  Generated:  {generated}")
            print(f"  Reference:  {case['reference_summary']}")
 
            if ROUGE_AVAILABLE:
                rouge = compute_rouge(generated, case["reference_summary"])
                rouge1_scores.append(rouge["rouge1"])
                rouge2_scores.append(rouge["rouge2"])
                rougeL_scores.append(rouge["rougeL"])
                interpretation = interpret_rouge(rouge["rouge1"])
                print(
                    f"  ROUGE-1: {rouge['rouge1']:.3f}  "
                    f"ROUGE-2: {rouge['rouge2']:.3f}  "
                    f"ROUGE-L: {rouge['rougeL']:.3f}  ({interpretation})"
                )
 
    # ============ Aggregate results ============
    print("\n" + "=" * 72)
    print("  AVERAGES")
    print("=" * 72)
 
    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0
    print(f"\n  Search metrics ({len(precisions)} queries):")
    print(f"    Precision@3: {avg_p:.3f}")
    print(f"    Recall@3:    {avg_r:.3f}")
 
    if rouge1_scores:
        avg_r1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_r2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rL = sum(rougeL_scores) / len(rougeL_scores)
        print(f"\n  Summary metrics ({len(rouge1_scores)} queries with reference):")
        print(f"    Avg ROUGE-1 F1: {avg_r1:.3f}  ({interpret_rouge(avg_r1)})")
        print(f"    Avg ROUGE-2 F1: {avg_r2:.3f}")
        print(f"    Avg ROUGE-L F1: {avg_rL:.3f}")
    elif ROUGE_AVAILABLE:
        print("\n  No reference summaries provided; skipping ROUGE.")
 
    print("\n" + "=" * 72)
    print("\nNext steps:")
    print("  - Paste these numbers into your reflection writeup")
    print("  - Note which queries failed and hypothesize why")
    print("  - Common failure modes:")
    print("    * Vocabulary mismatch (query words not in doc)")
    print("    * Chunk boundary cuts off the relevant info")
    print("    * Multiple docs cover the topic but only one is 'expected'")
    print("    * Summary hallucinates details not in the source chunks")
 
 
if __name__ == "__main__":
    evaluate()