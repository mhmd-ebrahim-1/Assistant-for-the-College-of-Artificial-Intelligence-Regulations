import json
import os
import pickle

import numpy as np

from rag_ksa_ai.text.processing import prepare_text


def build_index(chunks: list, index_dir: str = "./index") -> tuple:
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer

    os.makedirs(index_dir, exist_ok=True)
    texts = [prepare_text(c) for c in chunks]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=10000,
        sublinear_tf=True,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, f"{index_dir}/faiss.index")
    with open(f"{index_dir}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{index_dir}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Index built: {dim} dims, {index.ntotal} vectors")
    return index, vectorizer, chunks
