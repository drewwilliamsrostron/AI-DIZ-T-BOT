@echo off
REM ----------------------------------------------------------
REM Artibot launcher - run_artibot.bat
REM Double‑click this file to launch your bot.
REM It will:
REM   • Activate the local virtual environment (if present)
REM   • Fall back to the global Python interpreter otherwise
REM   • Pass any arguments you supply on the command line
REM ----------------------------------------------------------

SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT=%SCRIPT_DIR%run_artibot.py"

REM Detect & activate virtual environment
IF EXIST "%SCRIPT_DIR%venv\\Scripts\\activate.bat" (
    CALL "%SCRIPT_DIR%venv\\Scripts\\activate.bat"
)

python "%SCRIPT%" %*

PAUSE
