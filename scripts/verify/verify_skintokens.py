"""Verification script for SkinTokens environment.

Can be run directly via:
    uv run --python 3.11 verify_skintokens.py
or inside the activated venv:
    python verify_skintokens.py
"""

import sys

def verify_environment():
    print("=" * 60)
    print("         SkinTokens Environment Verification")
    print("=" * 60)
    
    # 1. Check Python version
    py_ver = sys.version_info
    print(f"[*] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if (py_ver.major, py_ver.minor) != (3, 11):
        print(f"[!] Warning: SkinTokens is tested with Python 3.11 (current: {py_ver.major}.{py_ver.minor})")
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

    # 3. Check HuggingFace Hub
    try:
        import huggingface_hub
        print(f"[+] huggingface_hub loaded successfully (v{huggingface_hub.__version__})")
    except ImportError as e:
        print(f"[-] huggingface_hub import error: {e}")

    # 4. Check core scientific libraries
    for pkg in ["numpy", "scipy", "einops"]:
        try:
            mod = __import__(pkg)
            print(f"[+] {pkg} loaded successfully (v{getattr(mod, '__version__', 'unknown')})")
        except ImportError as e:
            print(f"[-] {pkg} import warning: {e}")

    # 5. Check flash-attn (optional speedup)
    try:
        import flash_attn
        print(f"[+] flash-attn loaded successfully (v{flash_attn.__version__})")
    except ImportError:
        print("[*] flash-attn is not installed (optional, standard PyTorch attention will be used).")

    print("=" * 60)
    print("Verification complete.")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()