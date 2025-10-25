@echo off
set PROJECT_ENV_PATH=.\.venv

REM --- 1. Create the environment if it doesn't exist ---
if not exist %PROJECT_ENV_PATH% (
    echo Creating virtual environment...
    python -m venv %PROJECT_ENV_PATH%
    if errorlevel 1 goto :error
)

REM --- 2. Activate the environment (using the cmd/batch activate script) ---
echo Activating virtual environment...
call %PROJECT_ENV_PATH%\Scripts\activate.bat

REM --- 3. Install/Ensure dependencies are installed (you need a requirements.txt) ---
echo Installing dependencies (if needed)...
pip install -r requirements.txt

REM --- 4. Run the main application script ---
echo Running RunSound AI application...
python src\app.py

goto :eof

:error
echo.
echo ==========================================================
echo !!! ERROR: Failed to create or setup the environment. !!!
echo !!! Ensure Python is installed and accessible via "python".
echo ==========================================================
pause
:eof