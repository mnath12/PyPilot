#!/usr/bin/env python3
"""
Minimal test script to diagnose PyPilot issues
Run: python3 minimal_test.py
"""

print("🔍 PyPilot Minimal Test")
print("=" * 30)

# Test 1: Basic Python
print("\n1️⃣ Python: ", end="")
try:
    import sys
    print(f"✅ {sys.version.split()[0]}")
except Exception as e:
    print(f"❌ {e}")
    exit(1)

# Test 2: PyTorch
print("2️⃣ PyTorch: ", end="")
try:
    import torch
    print(f"✅ {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("❌ Not installed")
    print("   Fix: pip3 install torch")
    exit(1)
except Exception as e:
    print(f"❌ {e}")
    exit(1)

# Test 3: Transformers
print("3️⃣ Transformers: ", end="")
try:
    import transformers
    print(f"✅ {transformers.__version__}")
except ImportError:
    print("❌ Not installed")
    print("   Fix: pip3 install transformers")
    exit(1)
except Exception as e:
    print(f"❌ {e}")
    exit(1)

# Test 4: Datasets
print("4️⃣ Datasets: ", end="")
try:
    import datasets
    print(f"✅ {datasets.__version__}")
except ImportError:
    print("❌ Not installed")
    print("   Fix: pip3 install datasets")
    exit(1)
except Exception as e:
    print(f"❌ {e}")
    exit(1)

# Test 5: Simple model loading
print("5️⃣ Model Loading: ", end="")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("✅ GPT-2 tokenizer loaded")
except Exception as e:
    print(f"❌ {e}")
    print("   This might be a network issue")

print("\n🎯 DIAGNOSIS:")
print("=" * 30)
print("✅ All core packages working!")
print("✅ Ready to run PyPilot")
print("\nNext: python3 model.py --quick")