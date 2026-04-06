import json
import pickle


def load_index(index_dir: str = "./index") -> tuple:
    import faiss

    index = faiss.read_index(f"{index_dir}/faiss.index")
    with open(f"{index_dir}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{index_dir}/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded index: {index.ntotal} vectors")
    return index, vectorizer, chunks
