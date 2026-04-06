"""Build index from data files.

Run: python tools/build_index.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_ksa_ai import LaihaRAG
from rag_ksa_ai.config import DATA_FILES, INDEX_DIR


if __name__ == "__main__":
    rag = LaihaRAG(INDEX_DIR)
    rag.build_from_json(DATA_FILES)
    print("Index build completed.")
