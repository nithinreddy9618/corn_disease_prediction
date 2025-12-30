@echo off
REM Helper to run the Streamlit app using the current Python installation (avoids missing streamlit on PATH)
REM Usage: double-click or run from PowerShell/CMD: run_app.bat
cd /d "%~dp0app"
py -m streamlit run main.py %*
