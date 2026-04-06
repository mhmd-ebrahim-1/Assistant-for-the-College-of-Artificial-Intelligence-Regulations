# Quick local runner for the Flask app

& .\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python tools/build_index.py
python app/main.py
