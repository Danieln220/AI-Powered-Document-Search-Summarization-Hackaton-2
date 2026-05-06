# Search module: takes a natural-language query and returns the most
#relevant chunks from the index.

from embeddings import embed_texts, build_or_load_index
from config import TOP_K


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    
    index, metadata = build_or_load_index()
    
    
    if index.ntotal == 0:
        return []

    query_embedding = embed_texts([query])
    
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        
        if idx == -1:
            continue
        # Look up the human-readable info from metadata
        result = metadata[idx].copy()  
        result["score"] = float(score)  
        results.append(result)
    
    return results