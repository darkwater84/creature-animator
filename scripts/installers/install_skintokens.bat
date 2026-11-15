@echo off
setlocal enabledelayedexpansion
title SkinTokens Installer with uv (PyTorch 2.7.0 + CUDA 12.8)

echo ========================================================
echo     SkinTokens Automated Setup using Astral uv
echo     Target: D:\clean-implementation\engines\SkinTokens
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
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    )
    if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
)

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Falling back to installing uv via pip...
    python -m pip install --upgrade uv >nul 2>nul
)

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
echo [2/6] Ensuring Python 3.11 is available via uv ...
uv python install 3.11
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv failed to install Python 3.11.
    pause
    exit /b 1
)

echo.
echo [3/6] Setting up target directory and SkinTokens repository ...
if not exist "D:\clean-implementation\engines" (
    mkdir "D:\clean-implementation\engines"
)
cd /d "D:\clean-implementation\engines"

if exist "D:\clean-implementation\engines\SkinTokens\.git" (
    echo [INFO] SkinTokens repository already cloned. Pulling latest changes...
    cd SkinTokens
    git pull
) else (
    echo [INFO] Cloning SkinTokens repository...
    git clone https://github.com/VAST-AI-Research/SkinTokens.git
    cd SkinTokens
)

echo.
echo [4/6] Creating fresh Python 3.11 virtual environment ...
if not exist "venv" (
    uv venv --python 3.11 venv
)
call .\venv\Scripts\activate.bat

echo.
echo [5/6] Rapidly installing dependencies with uv pip ...
:: 1. PyTorch 2.7.0 with CUDA 12.8 (official SkinTokens spec)
echo [INFO] Installing PyTorch 2.7.0 (CUDA 12.8)...
uv pip install --python .\venv\Scripts\python.exe torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
if %ERRORLEVEL% neq 0 (
    echo [WARN] cu128 failed. Falling back to cu126...
    uv pip install --python .\venv\Scripts\python.exe torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
)

:: 2. Repository dependencies
if exist "requirements.txt" (
    uv pip install --python .\venv\Scripts\python.exe -r requirements.txt
)

:: 3. Utility packages & Hugging Face hub
uv pip install --python .\venv\Scripts\python.exe huggingface_hub numpy==1.26.4 einops scipy

:: 4. Optional model weights download
if exist "download.py" (
    echo [INFO] Downloading pretrained weights...
    .\venv\Scripts\python.exe download.py --model
)

echo.
echo [6/6] Verifying installed environment ...
(
echo import sys
echo print(f"Active Python: {sys.version}"^)
echo import torch
echo print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}"^)
echo import huggingface_hub
echo print(f"huggingface_hub loaded (v{huggingface_hub.__version__})"^)
echo print("SUCCESS: SkinTokens environment is fully ready!"^)
) > test_skintokens.py

.\venv\Scripts\python.exe test_skintokens.py

echo.
echo ========================================================
echo SkinTokens installation completed successfully!
echo Repository: D:\clean-implementation\engines\SkinTokens
echo Virtual environment: D:\clean-implementation\engines\SkinTokens\venv
echo ========================================================
echo.
pause
