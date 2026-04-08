import warnings
import glob

from tools.build_index import LaihaRAG, INDEX_DIR

warnings.warn(
    "`build_clean_index.py` is deprecated. Use `python tools/build_index.py`.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    # هات كل ملفات JSON من فولدر data
    json_files = glob.glob("data/*.json")

    if not json_files:
        raise FileNotFoundError("No JSON files found in data/ folder.")

    print("Found JSON files:", json_files)

    rag = LaihaRAG(INDEX_DIR)
    rag.build_from_json(json_files)

    print("Index build completed.")