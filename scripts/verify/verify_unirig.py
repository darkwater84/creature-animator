"""Verification script for UniRig environment.

Can be run directly via:
    uv run --python 3.11 verify_unirig.py
or inside the activated venv:
    python verify_unirig.py
"""

import sys

def verify_environment():
    print("=" * 60)
    print("           UniRig Environment Verification")
    print("=" * 60)
    
    # 1. Check Python version
    py_ver = sys.version_info
    print(f"[*] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if (py_ver.major, py_ver.minor) != (3, 11):
        print(f"[!] WARNING: Python 3.11 is required for UniRig! (Current: {py_ver.major}.{py_ver.minor})")
    else:
        print("[+] Python version check passed (3.11).")
        
    # 2. Check PyTorch & CUDA
    try:
        import torch
        print(f"[*] PyTorch Version: {torch.__version__}")
        print(f"[*] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[*] GPU Device: {torch.cuda.get_device_name(0)}")
        else:
            print("[!] Notice: CUDA is not available. GPU acceleration will be disabled.")
    except ImportError as e:
        print(f"[-] PyTorch missing: {e}")

    # 3. Check Blender bpy
    try:
        import bpy
        print(f"[+] bpy (Blender Python API) loaded successfully! Version: {bpy.app.version_string}")
    except ImportError as e:
        print(f"[-] bpy import error: {e}")

    # 4. Check spconv
    try:
        import spconv
        print(f"[+] spconv (Sparse Convolution) loaded successfully! Version: {spconv.__version__}")
    except ImportError as e:
        print(f"[-] spconv import error: {e}")

    # 5. Check HuggingFace Hub
    try:
        import huggingface_hub
        print("[+] huggingface_hub loaded successfully!")
    except ImportError as e:
        print(f"[-] huggingface_hub import error: {e}")

    print("=" * 60)
    print("Verification complete.")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()