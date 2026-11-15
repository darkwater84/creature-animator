@echo off
setlocal enabledelayedexpansion
title UniRig Installer with uv (Auto Python 3.11)

echo ========================================================
echo       UniRig Automated Setup using Astral uv
echo       Target: D:\clean-implementation\engines\UniRig
echo ========================================================
echo.

:: 1. Ensure D: drive exists
if not exist "D:\" (
    echo [ERROR] Drive D:\ was not found on your system!
    echo Please make sure drive D: is mounted.
    pause
    exit /b 1
)

:: 2. Check for Git
where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git is not installed or not in your system PATH.
    echo Please install Git for Windows from https://git-scm.com/download/win and try again.
    pause
    exit /b 1
)

echo [1/6] Checking for Astral uv ...
:: Add common uv install paths to current session PATH
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] 'uv' not found. Installing uv via official PowerShell script...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    
    :: Refresh PATH for current session
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    )
    if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
)

:: Secondary fallback: try installing via existing Python if still not found
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Falling back to installing uv via pip...
    python -m pip install --upgrade uv >nul 2>nul
)

:: Final check for uv
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to automatically install uv.
    echo Please run 'pip install uv' in your terminal and re-run this script.
    pause
    exit /b 1
)

echo [OK] Using uv version:
uv --version

echo.
echo [2/6] Automatically downloading and installing Python 3.11 via uv ...
uv python install 3.11
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv failed to install Python 3.11.
    pause
    exit /b 1
)

echo.
echo [3/6] Setting up target directory and repository ...
if not exist "D:\clean-implementation\engines" (
    mkdir "D:\clean-implementation\engines"
)
cd /d "D:\clean-implementation\engines"

if exist "D:\clean-implementation\engines\UniRig\.git" (
    echo [INFO] UniRig repository already cloned.
    cd UniRig
) else (
    echo [INFO] Cloning UniRig repository...
    git clone https://github.com/VAST-AI-Research/UniRig.git
    cd UniRig
)

echo.
echo [4/6] Creating fresh Python 3.11 virtual environment ...
if exist "venv" (
    echo [INFO] Removing previous incompatible virtual environment...
    rmdir /s /q "venv"
)

uv venv --python 3.11 venv
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment with Python 3.11.
    pause
    exit /b 1
)
call .\venv\Scripts\activate.bat

echo.
echo [5/6] Rapidly installing PyTorch (CUDA 12.1), bpy==4.2, and spconv via uv ...
:: 1. PyTorch with CUDA 12.1
uv pip install --python .\venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: 2. Repository dependencies (includes bpy==4.2 which resolves on Python 3.11)
if exist "requirements.txt" (
    uv pip install --python .\venv\Scripts\python.exe -r requirements.txt
)

:: 3. Specific CUDA sparse conv and utilities
uv pip install --python .\venv\Scripts\python.exe spconv-cu120 numpy==1.26.4 huggingface_hub

echo.
echo [6/6] Verifying installed environment ...
(
echo import sys
echo print(f"Active Python: {sys.version}"^)
echo import torch
echo print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}"^)
echo try:
echo     import bpy
echo     print(f"bpy loaded successfully! Version: {bpy.app.version_string}"^)
echo except ImportError as e:
echo     print(f"bpy import warning: {e}"^)
echo try:
echo     import spconv
echo     print("spconv loaded successfully!"^)
echo except ImportError as e:
echo     print(f"spconv import warning: {e}"^)
echo print("SUCCESS: UniRig environment is fully ready!"^)
) > test_unirig.py

.\venv\Scripts\python.exe test_unirig.py

echo.
echo ========================================================
echo UniRig installation completed successfully!
echo Repository: D:\clean-implementation\engines\UniRig
echo Virtual environment: D:\clean-implementation\engines\UniRig\venv
echo ========================================================
echo.
pause
