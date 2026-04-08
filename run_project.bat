@echo off
title AI RAG FINAL RUNNER

echo =====================================
echo Starting AI Project...
echo =====================================

REM ادخل على فولدر المشروع
cd /d %~dp0

echo.
echo [1] Activating venv...
call venv\Scripts\activate.bat

IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: venv not found!
    pause
    exit
)

echo.
echo [2] Starting Ollama...
start "Ollama" cmd /k "ollama serve"

timeout /t 4 >nul

echo.
echo [3] Checking model...
ollama list | findstr "qwen2.5:1.5b-instruct" >nul

IF %ERRORLEVEL% NEQ 0 (
    echo Downloading model...
    ollama pull qwen2.5:1.5b-instruct
)

echo.
echo [4] Building index...
python build_clean_index.py

echo.
echo [5] Starting Flask (IMPORTANT)...
start "Flask Server" cmd /k "cd /d %cd% && venv\Scripts\activate && python -m app.main && pause"

timeout /t 3 >nul

echo.
echo [6] Opening browser...
start http://127.0.0.1:5000

echo.
echo =====================================
echo DONE 🚀
echo =====================================

pause