@echo off
REM Startup script for Graph RAG PDF Chatbot on Windows

echo.
echo ========================================
echo Graph RAG PDF Chatbot - Startup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Check if .env exists
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create a .env file with your credentials.
    echo You can use .env.example as a template.
    echo.
    pause
)

REM Start Streamlit app
echo.
echo Starting Streamlit application...
echo Open http://localhost:8501 in your browser
echo.
streamlit run app.py

pause
