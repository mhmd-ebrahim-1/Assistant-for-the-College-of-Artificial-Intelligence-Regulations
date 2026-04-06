# AI Faculty Regulations Assistant (KFS University)

Local Arabic-first RAG assistant for Faculty of AI regulations and staff information at Kafr El-Sheikh University. The project is designed for transparent, source-grounded answers with optional local LLM phrasing.


## Value Proposition

1. Fast, local retrieval over official faculty data.
2. Better answer precision for Arabic staff and role-based queries.
3. Reliable fallback behavior: retrieval mode works even without Ollama.
4. Explainable output with source snippets and relevance scores.

## Prerequisites

1. Python 3.10 or newer.
2. pip and virtual environment support.
3. Optional: Ollama for AI phrasing mode.
4. Recommended Ollama model: `qwen2.5:1.5b-instruct`.

## Setup and Run

### Linux and macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/build_index.py
python app/main.py
```

### Windows PowerShell

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools/build_index.py
python app/main.py
```

Open `http://127.0.0.1:5000`.

Optional one-command run:

```powershell
.\run.ps1
```

## Architecture Overview

The codebase was refactored from a flat, monolithic layout into a layered structure:

```py
rag-ksu-ai/
│
├── app/                     # Web layer (Flask app)
│   ├── __init__.py
│   ├── main.py              # App factory / entry setup
│   ├── routes.py            # API routes / endpoints
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS, images
│
├── core/                    # Core RAG logic
│   ├── __init__.py
│   ├── config.py            # Settings & constants
│   ├── rag.py               # Main orchestrator
│   │
│   ├── data/
│   │   └── loader.py        # Load & normalize data
│   │
│   ├── processing/
│   │   ├── normalization.py
│   │   └── chunking.py      # Text splitting
│   │
│   ├── indexing/
│   │   ├── builder.py       # Build vector index
│   │   └── store.py         # Load/save index
│   │
│   ├── retrieval/
│   │   ├── hybrid.py        # Hybrid search logic
│   │   ├── scoring.py
│   │   └── filters.py
│   │
│   └── generation/
│       ├── llm.py           # LLM interface (Ollama/OpenAI/etc.)
│       └── formatter.py     # Output formatting
│
├── data/                    # Raw data (JSON, etc.)
│   ├── DrData.json
│   └── UniData.json
│
├── index/                   # Generated embeddings / FAISS
│
├── tools/                   # CLI & scripts
│   ├── build_index.py
│   └── cli.py
│
├── tests/                   # Unit & integration tests
│
├── requirements.txt
├── .env
├── README.md
└── run.py                   # Single entry point
```

### Runtime Flow

1. User submits a query via Flask UI.
2. Query is served from TTL cache if available.
3. Hybrid retrieval runs (`retrieval/hybrid.py`): TF-IDF cosine + keyword score + intent filter.
4. Answer strategy:
   - Deterministic formatting for staff and structured intents.
   - Ollama generation when available.
   - Retrieval-only fallback when Ollama is unavailable or fails.
5. Render answer and source cards in the template.

## Standardized Naming and Migration

### Renamed/Relocated

1. `flask_app.py` -> `app/main.py`.
2. `rag_system.py` -> modular package under `src/rag_ksa_ai/`.
3. `build_clean_index.py` -> `tools/build_index.py`.

### Compatibility Policy

Legacy files are currently preserved as shims for one release cycle:

1. `python flask_app.py`
2. `python rag_system.py` (imports only)
3. `python build_clean_index.py`

Use new entrypoints for all new development.

## Common Commands

1. Build index: `python tools/build_index.py`
2. Run web app: `python app/main.py`
3. Run CLI: `python tools/cli.py`
4. Syntax check key modules:

```bash
python -m py_compile app/main.py app/routes.py src/rag_ksa_ai/rag.py
```

## Contribution Guidelines

1. Read `CONTRIBUTING.md` before opening a PR.
2. Keep changes scoped by layer (`data`, `retrieval`, `generation`, `app`).
3. Add or update tests and manual validation notes for behavior changes.
4. Update README and migration notes when commands or structure change.
5. Do not add new logic to deprecated compatibility shims.

## Troubleshooting

1. Stale answers after editing JSON files:
   - Rebuild index with `python tools/build_index.py`.
2. Ollama unavailable:
   - App remains operational in retrieval-only mode.
3. Import errors for `rag_ksa_ai`:
   - Run from repository root, or install in editable mode:

```bash
pip install -e .
```

## License

Add a `LICENSE` file before public open-source release.
