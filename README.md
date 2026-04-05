# AI Faculty Regulations Assistant (KFS University)

A local RAG web app for answering questions about the Faculty of Artificial Intelligence regulations at Kafr El-Sheikh University, with clear source snippets and improved accuracy for faculty/staff questions.

## Overview

The system combines:
- Hybrid retrieval: FAISS + TF-IDF + keyword boosting
- Multi-source indexing from:
	- `data.json` (regulations and academic rules)
	- `data2.json` (faculty/staff and administration data)
- Optional local generation with Ollama (`qwen2.5:1.5b-instruct`)
- Deterministic (non-LLM) answers for staff profile intents (email, role, specialization)

## Requirements

- Python 3.13+
- Windows PowerShell (commands below are Windows-friendly)
- Optional: Ollama for higher quality answer phrasing
	- Model: `qwen2.5:1.5b-instruct`

## Run (Windows)

1. Activate the virtual environment:

```powershell
& .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
d:/Downloads/files/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

3. Build/rebuild index (from `data.json` + `data2.json`):

```powershell
d:/Downloads/files/.venv/Scripts/python.exe build_clean_index.py
```

4. Start Ollama (optional):

```powershell
ollama serve
```

5. Start Flask app:

```powershell
d:/Downloads/files/.venv/Scripts/python.exe flask_app.py
```

Open:
- `http://127.0.0.1:5000`

## One-Command Run

```powershell
.\run.ps1
```

## Project Structure

```text
files/
|-- flask_app.py
|-- rag_system.py
|-- build_clean_index.py
|-- data.json
|-- data2.json
|-- requirements.txt
|-- run.ps1
|-- templates/
|   \-- index.html
|-- static/
|   |-- style.css
|   |-- app.js
|   |-- favicon.svg
|   |-- faculty-logo.png
|   \-- university-logo.png
\-- index/
		|-- faiss.index
		|-- vectorizer.pkl
		\-- chunks.json
```

## Current Features

- Fast local semantic retrieval over faculty regulations
- Staff-aware question handling (dean/vice-dean/secretary/lecturer/profile)
- Deterministic answers for high-precision staff intents:
	- email queries
	- specialization queries
	- leadership role queries
- Smart filtering for course plan queries by level/semester
- Automatic fallback to retrieval when Ollama is unavailable
- In-memory caching for repeated questions and Ollama status
- UI with suggested questions, source cards, and logo branding

## Latest Update (April 2026)

- Expanded `data2.json` ingestion to support nested structures:
	- faculty leadership blocks
	- dean full profile
	- departments with member lists
	- administrative staff
	- summary statistics
- Improved staff answering quality:
	- richer profile responses (role, specialization, dates, achievements, interests, memberships)
	- better Arabic name matching and handling for short-form queries
	- disambiguation suggestions when multiple staff names are likely
- Improved statistics and routing logic:
	- aggregate count questions are prioritized to statistics sources
	- source cards are now intent-sorted with a reason shown for why each source was selected
- Improved UX/UI:
	- categorized suggested-question tabs (Regulations / Staff / Statistics)
	- dynamic "More Questions" button (`/api/suggest-more`) generated from indexed content
	- enhanced logo presentation and cleaner answer panel layout
- Removed short-answer summary block to keep responses fully detailed

## Quick QA Checklist

- Rebuild index after editing `data.json` or `data2.json`:

```powershell
d:/Downloads/files/.venv/Scripts/python.exe build_clean_index.py
```

- Syntax check backend files:

```powershell
d:/Downloads/files/.venv/Scripts/python.exe -m py_compile flask_app.py rag_system.py
```

## Example Queries

### Regulations
- How many credit hours are required for graduation?
- What are the honor degree conditions?
- What is the passing grade?

### Staff and Administration
- Who is the vice dean for education and students?
- What is Dr. Mahmoud Yassein Shams' specialization?
- What is Dr. Tamer Medhat's email?
- How many faculty staff members are there?

## Troubleshooting

### App does not open
- Ensure port `5000` is not occupied by another process.
- Restart with:

```powershell
d:/Downloads/files/.venv/Scripts/python.exe flask_app.py
```

### Answers are stale after editing JSON files
- Rebuild index:

```powershell
d:/Downloads/files/.venv/Scripts/python.exe build_clean_index.py
```

### Ollama start fails with port-in-use error
- Ollama may already be running on `11434`.
- You can still run the app; it will connect if Ollama is already active.

## Notes

- Local-first architecture (no external API required for retrieval).
- Works without Ollama (retrieval-only mode).
- For best results, ask direct questions with clear keywords.
