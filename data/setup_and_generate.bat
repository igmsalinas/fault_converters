@REM  ============================================================================
@REM  setup_and_generate.bat — One-shot Windows environment setup + data generation
@REM  ============================================================================
@REM
@REM  USAGE (from PowerShell, any CWD):
@REM
@REM    & "\\wsl.localhost\Ubuntu\home\nacho\projects\fault_converters\data\setup_and_generate.bat"
@REM
@REM  Or pass arguments to generate_data.py:
@REM
@REM    & "...\setup_and_generate.bat" --converter buck --dataset-name dataset_01 --n-normal 10000 --n-fault 2000
@REM    & "...\setup_and_generate.bat" --converter buck --dataset-name dataset_01 --n-normal 10000 --n-fault 2000 --estimate
@REM
@REM  WHAT IT DOES:
@REM    1. Locates or creates .venv_win (Python 3.12 venv inside the repo)
@REM    2. Installs psimapipy from the Altair PSIM wheel (with .pyc→.py source fix)
@REM    3. Installs numpy (the only third-party dependency for generation)
@REM    4. Runs data/generate_data.py with any arguments you pass
@REM  ============================================================================

@echo off
setlocal EnableDelayedExpansion

:: ---- Resolve paths relative to this script (repo_root/data/) ----------------
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

:: Repo root is one level up from data/
for %%A in ("!SCRIPT_DIR!") do set "REPO_ROOT=%%~dpA"
if "!REPO_ROOT:~-1!"=="\" set "REPO_ROOT=!REPO_ROOT:~0,-1!"

echo ============================================================
echo  PSIM Data Generation — Environment Setup
echo ============================================================
echo  Repo root : !REPO_ROOT!
echo  Script dir: !SCRIPT_DIR!
echo ============================================================

:: ---- Locate Python 3.12 -----------------------------------------------------
set "PYTHON_EXE="

:: Option 1: py launcher (most reliable)
where py >nul 2>&1
if !ERRORLEVEL! equ 0 (
    for /f "tokens=*" %%I in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
    )
)

:: Option 2: Check common install paths
if "!PYTHON_EXE!"=="" (
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%PROGRAMFILES%\Python312\python.exe"
        "%PROGRAMFILES(x86)%\Python312\python.exe"
        "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    ) do (
        if exist %%P (
            set "PYTHON_EXE=%%~P"
            goto :FOUND_PYTHON
        )
    )
)

:: Option 3: Fall back to any python on PATH
if "!PYTHON_EXE!"=="" (
    where python >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        for /f "tokens=*" %%I in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
            set "PYTHON_EXE=%%I"
        )
    )
)

:FOUND_PYTHON
if "!PYTHON_EXE!"=="" (
    echo ERROR: Python not found. Install Python 3.12 and add to PATH.
    echo        Download: https://www.python.org/downloads/release/python-3129/
    exit /b 1
)

echo Found Python: !PYTHON_EXE!
"!PYTHON_EXE!" --version

:: ---- Create or reuse .venv_win ----------------------------------------------
set "VENV_DIR=!REPO_ROOT!\.venv_win"
set "VENV_PYTHON=!VENV_DIR!\Scripts\python.exe"
set "VENV_PIP=!VENV_DIR!\Scripts\pip.exe"

if not exist "!VENV_PYTHON!" (
    echo.
    echo Creating Windows virtual environment at !VENV_DIR! ...
    "!PYTHON_EXE!" -m venv "!VENV_DIR!"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists at !VENV_DIR!
)

:: ---- Upgrade pip ------------------------------------------------------------
"!VENV_PYTHON!" -m pip install --upgrade pip --quiet

:: ---- Install numpy ----------------------------------------------------------
echo.
echo Installing numpy...
"!VENV_PIP!" install numpy --quiet
if !ERRORLEVEL! neq 0 (
    echo ERROR: Failed to install numpy.
    exit /b 1
)

:: ---- Install psimapipy ------------------------------------------------------
echo.
echo Installing psimapipy...

:: PSIM default installation path
set "PSIM_WHEEL_DIR=C:\Altair\Altair_PSIM_2026\Python\Source"
set "PSIM_WHEEL=!PSIM_WHEEL_DIR!\dist\psimapipy-2026.0-py3-none-any.whl"
set "PSIM_SOURCE=!PSIM_WHEEL_DIR!\psimapipy-sources\Psim_Class.py"

:: Also check repo-local copy
set "LOCAL_WHEEL=!REPO_ROOT!\data\psimapipy-2026.0-py3-none-any.whl"

if exist "!PSIM_WHEEL!" (
    "!VENV_PIP!" install "!PSIM_WHEEL!" --force-reinstall --quiet 2>nul
) else if exist "!LOCAL_WHEEL!" (
    "!VENV_PIP!" install "!LOCAL_WHEEL!" --force-reinstall --quiet 2>nul
) else (
    echo WARNING: psimapipy wheel not found at:
    echo   !PSIM_WHEEL!
    echo   !LOCAL_WHEEL!
    echo Skipping psimapipy install — PSIM simulations will fail.
    goto :SKIP_PSIM_FIX
)

:: ---- Fix .pyc magic number mismatch ----------------------------------------
:: The Altair wheel ships a .pyc compiled with a specific Python version.
:: If versions mismatch, replace .pyc with the original .py source.
set "SITE_PKG=!VENV_DIR!\Lib\site-packages\psimapipy"

"!VENV_PYTHON!" -c "import psimapipy" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo Fixing psimapipy .pyc version mismatch...
    if exist "!PSIM_SOURCE!" (
        copy /y "!PSIM_SOURCE!" "!SITE_PKG!\Psim_Class.py" >nul
        if exist "!SITE_PKG!\Psim_Class.pyc" del "!SITE_PKG!\Psim_Class.pyc"
        echo Fixed: copied Psim_Class.py source over incompatible .pyc
    ) else (
        echo ERROR: Cannot fix psimapipy — source file not found at:
        echo   !PSIM_SOURCE!
        echo   Please rebuild the wheel using the BAT in !PSIM_WHEEL_DIR!
        exit /b 1
    )
)

:: Verify psimapipy works
"!VENV_PYTHON!" -c "import psimapipy; print('  psimapipy OK')"
if !ERRORLEVEL! neq 0 (
    echo ERROR: psimapipy import still fails after fix attempt.
    exit /b 1
)

:SKIP_PSIM_FIX

:: ---- Run generate_data.py ---------------------------------------------------
echo.
echo ============================================================
echo  Running data generation...
echo ============================================================
echo.

set "GENERATE_SCRIPT=!SCRIPT_DIR!\generate_data.py"

:: Pass all script arguments through to generate_data.py
"!VENV_PYTHON!" "!GENERATE_SCRIPT!" %*

echo.
echo ============================================================
echo  Done.
echo ============================================================
endlocal
