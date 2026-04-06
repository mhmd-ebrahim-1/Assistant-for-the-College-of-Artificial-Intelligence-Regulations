"""Compatibility module.

Deprecated: import from `rag_ksa_ai` package under `src/`.
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_ksa_ai import (  # noqa: E402
    LaihaRAG,
    OLLAMA_MODEL,
    check_ollama,
    compose_staff_answer,
    generate_answer,
    is_staff_query,
)
from rag_ksa_ai.data.loader import load_json_data, normalize_data_records  # noqa: E402
from rag_ksa_ai.retrieval.hybrid import retrieve  # noqa: E402
from rag_ksa_ai.text.processing import prepare_text  # noqa: E402

warnings.warn(
    "`rag_system.py` is deprecated. Import from `rag_ksa_ai` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "LaihaRAG",
    "OLLAMA_MODEL",
    "check_ollama",
    "generate_answer",
    "compose_staff_answer",
    "is_staff_query",
    "load_json_data",
    "normalize_data_records",
    "prepare_text",
    "retrieve",
]
