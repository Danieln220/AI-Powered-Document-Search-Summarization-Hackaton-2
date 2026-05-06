# Generate vectors and manage the FAISS index

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, INDEX_FILE, METADATA_FILE

# Load the model once at module level (singleton pattern - avoids reloading)
print("Loading embedding model...")
_model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded.")


def embed_texts(texts: list[str], batch_size: int = 4) -> np.ndarray:
    #Embed a list of texts. Small batch size keeps CPU memory low.
    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # IMPORTANT: lets us use inner-product = cosine similarity
    )
    return embeddings


def build_or_load_index(dimension: int = 384):
    #Create a fresh FAISS index or load existing one from disk.
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    # IndexFlatIP = brute-force inner product. Simple, accurate, no training needed.
    # Perfect for a hackathon with a few hundred chunks.
    index = faiss.IndexFlatIP(dimension)
    metadata = []  # list of dicts: {"doc_name": ..., "chunk_index": ..., "text": ...}
    return index, metadata


def add_document_to_index(doc_name: str, chunks: list[str]):
    #Embed chunks and add them to the index, then save to disk.
    index, metadata = build_or_load_index()

    embeddings = embed_texts(chunks)
    index.add(embeddings)

    # Track which document and chunk each vector came from
    for i, chunk_text in enumerate(chunks):
        metadata.append({
            "doc_name": doc_name,
            "chunk_index": i,
            "text": chunk_text,
        })

    # Persist
    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    return len(chunks)