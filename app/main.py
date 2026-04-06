import os
import sys
from pathlib import Path

from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_ksa_ai import LaihaRAG
from rag_ksa_ai.config import ANSWER_CACHE_TTL, DATA_FILES, INDEX_DIR, OLLAMA_STATUS_TTL

from .cache import TTLCache
from .routes import bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

    rag = LaihaRAG(INDEX_DIR)
    rag.ensure_index(DATA_FILES)

    app.config["RAG_ENGINE"] = rag
    app.config["OLLAMA_CACHE"] = TTLCache(OLLAMA_STATUS_TTL)
    app.config["ANSWER_CACHE"] = TTLCache(ANSWER_CACHE_TTL)

    app.register_blueprint(bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
