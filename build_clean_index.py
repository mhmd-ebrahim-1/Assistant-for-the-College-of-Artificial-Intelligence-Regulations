"""Compatibility entrypoint.

Deprecated: use `python tools/build_index.py`.
"""

import warnings

from tools.build_index import LaihaRAG
from tools.build_index import DATA_FILES, INDEX_DIR

warnings.warn(
    "`build_clean_index.py` is deprecated. Use `python tools/build_index.py`.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    rag = LaihaRAG(INDEX_DIR)
    rag.build_from_json(DATA_FILES)
    print("Index build completed.")
