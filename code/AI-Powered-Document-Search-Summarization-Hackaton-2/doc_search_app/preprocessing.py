# Preprocessing : Split text into searchable chunks
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    #Normalize whitespace and remove obvious junk.
    # Collapse multiple newlines/spaces into single spaces
    text = re.sub(r"\s+", " ", text)
    # Remove very common PDF artifacts (page numbers like "Page 3 of 10")
    text = re.sub(r"Page \d+ of \d+", "", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP) -> list[str]:
    #Split text into chunks of roughly `chunk_size` words with `overlap` words
    #between consecutive chunks.
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap

    return chunks


def process_document(text: str) -> list[str]:
    #Full pipeline: clean then chunk.
    cleaned = clean_text(text)
    return split_into_chunks(cleaned)