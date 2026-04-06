"""Interactive CLI for local RAG queries."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_ksa_ai import LaihaRAG, OLLAMA_MODEL, check_ollama
from rag_ksa_ai.config import DATA_FILES, INDEX_DIR


if __name__ == "__main__":
    rag = LaihaRAG(INDEX_DIR)
    rag.ensure_index(DATA_FILES)

    ollama_ok = check_ollama()
    print(f"\n{'='*60}\nAI Faculty Regulations Assistant")
    print(f"Model: {OLLAMA_MODEL if ollama_ok else 'Retrieval only'}")
    print(f"{'='*60}\n")

    while True:
        q = input("Query: ").strip()
        if q in ("exit", "quit", "q"):
            break
        if not q:
            continue
        if ollama_ok:
            r = rag.ask(q)
            print(f"\nAnswer:\n{r['answer']}\n")
        else:
            print(rag.ask_no_llm(q))
