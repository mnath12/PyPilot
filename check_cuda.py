"""
Diagnostic script to check PyTorch CUDA installation and GPU availability.
"""

import sys
import subprocess

print("="*60)
print("PyTorch CUDA Diagnostic Tool")
print("="*60)

# Check PyTorch version
try:
    import torch
    print(f"\n✓ PyTorch version: {torch.__version__}")
except ImportError:
    print("\n✗ PyTorch is not installed!")
    sys.exit(1)

# Check CUDA availability in PyTorch
print(f"\nCUDA available in PyTorch: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Test bf16 support
        try:
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=f"cuda:{i}")
            print(f"  bf16 support: ✓")
        except Exception as e:
            print(f"  bf16 support: ✗ ({e})")
else:
    print("\n✗ CUDA is not available in PyTorch!")
    print("\nPossible causes:")
    print("1. PyTorch was installed without CUDA support (CPU-only version)")
    print("2. CUDA drivers are not installed")
    print("3. CUDA version mismatch between PyTorch and system")
    
    # Check if CUDA is installed on system
    print("\n" + "="*60)
    print("Checking system CUDA installation...")
    print("="*60)
    
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✓ nvcc (CUDA compiler) is available:")
            print(result.stdout.split('\n')[0] if result.stdout else "Version info found")
        else:
            print("\n✗ nvcc not found in PATH")
    except FileNotFoundError:
        print("\n✗ nvcc not found - CUDA toolkit may not be installed")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✓ nvidia-smi output:")
            print(result.stdout)
        else:
            print("\n✗ nvidia-smi failed")
    except FileNotFoundError:
        print("\n✗ nvidia-smi not found - NVIDIA drivers may not be installed")
    
    print("\n" + "="*60)
    print("Recommended Solutions:")
    print("="*60)
    print("\n1. Check if you have the CUDA-enabled PyTorch:")
    print("   Run: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("\n2. If False, reinstall PyTorch with CUDA support:")
    print("   Visit: https://pytorch.org/get-started/locally/")
    print("   Or use: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   (Replace cu121 with your CUDA version: cu118, cu121, etc.)")
    print("\n3. Verify NVIDIA drivers are installed:")
    print("   Run: nvidia-smi")
    print("\n4. Check CUDA toolkit installation:")
    print("   Run: nvcc --version")

print("\n" + "="*60)

