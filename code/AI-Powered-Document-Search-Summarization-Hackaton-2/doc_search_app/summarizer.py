#Summarization module: takes the chunks returned by search and produces
#a concise summary that directly answers the query (or at least covers
#the relevant ground).

from transformers import pipeline
from config import SUMMARIZATION_MODEL

print("Loading summarization model... (downloads ~500MB on first run)")
_summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)
print("Summarization model loaded.")

def summarize_chunks(chunks: list[str], max_length : int = 150, min_length: int = 40) -> str:
    if not chunks:
        return "No relevant content found."
    
    combined = " ".join(chunks)

    max_chars = 3500
    if len(combined) > max_chars:
        combined = combined[:max_chars]

    result = _summarizer(
        combined,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )

    return result[0]["summary_text"]