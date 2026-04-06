# Contributing Guide

## Development Setup

1. Create a virtual environment and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Rebuild index artifacts after data changes: `python tools/build_index.py`.
4. Run app locally: `python app/main.py`.

## Branch and PR Workflow

1. Create a feature branch from the default branch.
2. Keep each PR focused on one concern (retrieval, UI, docs, tooling).
3. Include a clear PR description with before/after behavior.
4. Attach manual test notes for representative queries.

## Code Style

1. Use snake_case for file and module names.
2. Keep logic in the correct layer:
   - `src/rag_ksa_ai/data` for data normalization.
   - `src/rag_ksa_ai/retrieval` for scoring and filtering.
   - `src/rag_ksa_ai/generation` for answer generation.
   - `app` for Flask routes and cache wiring.
3. Avoid adding new logic to compatibility shims (`rag_system.py`, `flask_app.py`, `build_clean_index.py`).

## Testing Expectations

1. Run syntax checks before opening a PR.
2. Validate both runtime modes:
   - Retrieval-only mode (without Ollama)
   - AI + retrieval mode (with Ollama)
3. Verify these query categories at minimum:
   - Regulations
   - Courses (level/semester)
   - Staff profile and contact
   - Statistics/count queries

## Documentation Expectations

1. Update README when commands, architecture, or structure change.
2. Add migration notes if old entrypoints or module paths are deprecated.
