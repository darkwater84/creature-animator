@echo off
setlocal enabledelayedexpansion
title DINOv2 Installer for Windows

echo ========================================================
echo     DINOv2 Setup for D:\clean-implementation\engines
echo ========================================================
echo.

:: 1. Ensure D: drive exists
if not exist "D:\" (
    echo [ERROR] Drive D:\ was not found on your system!
    echo Please make sure drive D: is mounted or update the drive letter.
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

:: 3. Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/ and check "Add Python to PATH".
    pause
    exit /b 1
)

echo [1/5] Creating directory D:\clean-implementation\engines ...
if not exist "D:\clean-implementation\engines" (
    mkdir "D:\clean-implementation\engines"
)
cd /d "D:\clean-implementation\engines"

echo [2/5] Cloning DINOv2 repository ...
if exist "D:\clean-implementation\engines\dinov2\.git" (
    echo DINOv2 repository already cloned. Pulling latest updates...
    cd dinov2
    git pull
) else (
    git clone https://github.com/facebookresearch/dinov2.git
    cd dinov2
)

echo.
echo [3/5] Setting up Python virtual environment ...
if not exist "venv" (
    python -m venv venv
)
call .\venv\Scripts\activate.bat

echo.
echo [4/5] Installing PyTorch with CUDA support and dependencies ...
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo.
echo [5/5] Creating verification script and testing ...
(
echo import torch
echo print("Loading DINOv2 ViT-S/14 via torch.hub..."^)
echo model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"^)
echo model.eval(^)
echo dummy_input = torch.randn(1, 3, 224, 224^)
echo with torch.inference_mode(^):
echo     features = model(dummy_input^)
echo print(f"SUCCESS: DINOv2 loaded successfully! Output feature shape: {features.shape}"^)
) > test_dinov2.py

python test_dinov2.py

echo.
echo ========================================================
echo DINOv2 installation and test completed!
echo Repository: D:\clean-implementation\engines\dinov2
echo Virtual environment: D:\clean-implementation\engines\dinov2\venv
echo ========================================================
echo.
pause
