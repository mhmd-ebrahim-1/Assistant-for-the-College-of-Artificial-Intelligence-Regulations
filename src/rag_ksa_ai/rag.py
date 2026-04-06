from pathlib import Path

from rag_ksa_ai.data.loader import load_json_data
from rag_ksa_ai.indexing.builder import build_index
from rag_ksa_ai.indexing.store import load_index
from rag_ksa_ai.retrieval.hybrid import retrieve
from rag_ksa_ai.generation.ollama import generate_answer


class LaihaRAG:
    def __init__(self, index_dir: str = "./index"):
        self.index_dir = index_dir
        self.index = self.vectorizer = self.chunks = None

    def ensure_index(self, json_path="data.json"):
        if isinstance(json_path, (list, tuple)):
            json_files = [Path(p) for p in json_path]
        else:
            json_files = [Path(json_path)]

        index_files = [
            Path(self.index_dir) / "faiss.index",
            Path(self.index_dir) / "vectorizer.pkl",
            Path(self.index_dir) / "chunks.json",
        ]

        if not all(p.exists() for p in index_files):
            self.build_from_json([str(p) for p in json_files])
            return

        index_mtime = min(p.stat().st_mtime for p in index_files)
        for data_file in json_files:
            if data_file.exists() and data_file.stat().st_mtime > index_mtime:
                self.build_from_json([str(p) for p in json_files])
                return

        self.load()

    def build_from_json(self, json_path="data.json"):
        if isinstance(json_path, (list, tuple)):
            json_paths = list(json_path)
        else:
            json_paths = [json_path]

        chunks = []
        for one_path in json_paths:
            if not Path(one_path).exists():
                continue
            print(f"Loading JSON: {one_path}")
            chunks.extend(load_json_data(one_path))

        if not chunks:
            raise FileNotFoundError("No JSON data files were found to build the index.")

        self.index, self.vectorizer, self.chunks = build_index(chunks, self.index_dir)
        print("RAG system ready from JSON!\n")

    def load(self):
        self.index, self.vectorizer, self.chunks = load_index(self.index_dir)

    def search(self, query: str, top_k: int = 5) -> list:
        if self.index is None:
            raise RuntimeError("Call build() or load() first.")
        return retrieve(query, self.index, self.vectorizer, self.chunks, top_k)

    def ask(self, query: str, top_k: int = 5, **kwargs) -> dict:
        retrieved = self.search(query, top_k)
        answer = generate_answer(query, retrieved)
        return {
            "query": query,
            "answer": answer,
            "sources": [{
                "page": c.get("page", "-"),
                "score": c.get("score", 0.0),
                "text": c.get("text") or c.get("text_ar") or c.get("description_en", ""),
                "preview": (c.get("text") or c.get("text_ar") or c.get("description_en", ""))[:120] + "...",
            } for c in retrieved],
        }

    def ask_no_llm(self, query: str, top_k: int = 3) -> str:
        retrieved = self.search(query, top_k)
        out = f"نتائج: '{query}'\n{'='*50}\n\n"
        for c in retrieved:
            page = c.get("page", "-")
            text = c.get("text") or c.get("text_ar") or c.get("description_en", "")
            out += f"صفحة {page} | تطابق: {c['score']:.3f}\n{text}\n\n{'─'*40}\n\n"
        return out
