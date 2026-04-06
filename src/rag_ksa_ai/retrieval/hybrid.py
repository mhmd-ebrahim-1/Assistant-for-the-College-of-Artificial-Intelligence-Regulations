import numpy as np

from rag_ksa_ai.retrieval.filters import rerank_staff_results, smart_filter
from rag_ksa_ai.retrieval.scoring import is_staff_query, keyword_score
from rag_ksa_ai.text.processing import prepare_text


def retrieve(query: str, index, vectorizer, chunks: list, top_k: int = 5) -> list:
    q_vec = vectorizer.transform([query]).toarray().astype(np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm
    scores, indices = index.search(q_vec, min(top_k * 3, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx].copy()
        tfidf_s = float(score)
        search_text = prepare_text(chunk)
        kw_s = keyword_score(query, search_text)
        chunk["score"] = round(tfidf_s * 0.6 + kw_s * 0.4, 4)
        results.append(chunk)

    results = smart_filter(results, query)

    if is_staff_query(query):
        results = rerank_staff_results(results, query)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
