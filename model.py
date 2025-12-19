#!/usr/bin/env python3
"""
PyPilot Model Manager
Downloads, saves, and manages code generation models

Usage:
    python model.py                          # Download default Qwen2.5-Coder
    python model.py --model deepseek-coder   # Download specific model
    python model.py --list                   # Show available models
    python model.py --test                   # Test downloaded model
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from datasets import load_dataset
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install torch transformers datasets")
    exit(1)

# Model configurations
MODEL_CONFIGS = {
    # Primary choice - Qwen2.5-Coder series
    "qwen2.5-coder": {
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "description": "Qwen 2.5 Coder 7B - Primary choice for code generation",
        "size": "7B",
        "type": "instruct",
        "context_length": 32768,
        "recommended": True
    },
    "qwen2.5-coder-1.5b": {
        "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", 
        "description": "Qwen 2.5 Coder 1.5B - Smaller, faster variant",
        "size": "1.5B",
        "type": "instruct",
        "context_length": 32768,
        "recommended": False
    },
    "qwen2.5-coder-3b": {
        "name": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "description": "Qwen 2.5 Coder 3B - Balanced size/performance",
        "size": "3B", 
        "type": "instruct",
        "context_length": 32768,
        "recommended": False
    },
    
    # DeepSeek alternatives
    "deepseek-coder": {
        "name": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "description": "DeepSeek Coder 6.7B - Strong code generation",
        "size": "6.7B",
        "type": "instruct", 
        "context_length": 16384,
        "recommended": False
    },
    "deepseek-coder-1.3b": {
        "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "description": "DeepSeek Coder 1.3B - Lightweight option",
        "size": "1.3B",
        "type": "instruct",
        "context_length": 16384,
        "recommended": False
    },
    
    # Microsoft options
    "codegen": {
        "name": "microsoft/CodeGPT-small-py",
        "description": "Microsoft CodeGPT - Python focused",
        "size": "117M",
        "type": "base",
        "context_length": 1024,
        "recommended": False
    },
    
    # Fallback options (smaller, always work)
    "diaglogpt": {
        "name": "microsoft/DialoGPT-small",
        "description": "DialoGPT Small - Reliable fallback",
        "size": "117M",
        "type": "base", 
        "context_length": 1024,
        "recommended": False
    },
    "gpt2": {
        "name": "gpt2",
        "description": "GPT-2 - Universal fallback",
        "size": "117M",
        "type": "base",
        "context_length": 1024,
        "recommended": False
    }
}

class ModelManager:
    """Manages downloading, saving, and loading of code generation models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"PyPilot Model Manager initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Models directory: {self.models_dir.absolute()}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    
    def list_available_models(self) -> None:
        """Display all available model configurations"""
        print("\n🤖 AVAILABLE MODELS")
        print("=" * 60)
        
        for key, config in MODEL_CONFIGS.items():
            status = "⭐ RECOMMENDED" if config.get("recommended") else "  "
            print(f"{status} {key}")
            print(f"    Name: {config['name']}")
            print(f"    Size: {config['size']}")
            print(f"    Type: {config['type']}")
            print(f"    Context: {config['context_length']} tokens")
            print(f"    Description: {config['description']}")
            print()
    
    def get_model_config(self, model_key: str) -> Dict[str, Any]:
        """Get configuration for a model"""
        if model_key not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_key}")
            logger.info("Available models:")
            for key in MODEL_CONFIGS.keys():
                logger.info(f"  - {key}")
            raise ValueError(f"Model '{model_key}' not found")
        
        return MODEL_CONFIGS[model_key]
    
    def download_model(self, model_key: str = "qwen2.5-coder", force: bool = False) -> str:
        """
        Download and save a model to disk
        
        Args:
            model_key: Key from MODEL_CONFIGS 
            force: Re-download even if already exists
            
        Returns:
            Path to saved model
        """
        config = self.get_model_config(model_key)
        model_name = config["name"]
        
        # Create model-specific directory
        model_dir = self.models_dir / model_key
        
        # Check if already downloaded
        if model_dir.exists() and not force:
            logger.info(f"✅ Model {model_key} already exists at {model_dir}")
            return str(model_dir)
        
        logger.info(f"📥 Downloading {model_key}: {model_name}")
        logger.info(f"📝 Description: {config['description']}")
        logger.info(f"💾 Size: {config['size']}")
        
        try:
            # Create directory
            model_dir.mkdir(exist_ok=True)
            
            # Download tokenizer
            logger.info("📥 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Download model
            logger.info("📥 Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None,  # Don't auto-load to GPU yet
                trust_remote_code=True
            )
            
            # Save to disk
            logger.info(f"💾 Saving to {model_dir}...")
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            
            # Save metadata
            metadata = {
                "model_key": model_key,
                "original_name": model_name,
                "config": config,
                "download_time": time.time(),
                "vocab_size": len(tokenizer),
                "device_used": str(self.device)
            }
            
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Successfully downloaded {model_key}")
            logger.info(f"📁 Saved to: {model_dir.absolute()}")
            
            return str(model_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_key}: {e}")
            
            # Try fallback if this was the primary choice
            if model_key == "qwen2.5-coder":
                logger.info("🔄 Trying fallback model...")
                return self.download_model("diaglogpt", force)
            else:
                raise e
    
    def load_model(self, model_key: str):
        """Load a downloaded model"""
        model_dir = self.models_dir / model_key
        
        if not model_dir.exists():
            logger.error(f"Model {model_key} not found. Download it first.")
            return None, None
        
        logger.info(f"📂 Loading {model_key} from {model_dir}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info(f"✅ Loaded {model_key} successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_key}: {e}")
            return None, None
    
    def test_model(self, model_key: str) -> None:
        """Test a downloaded model with a simple code generation task"""
        logger.info(f"🧪 Testing {model_key}...")
        
        model, tokenizer = self.load_model(model_key)
        if model is None:
            logger.error("Cannot test - model failed to load")
            return
        
        # Create pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Test prompts
        test_prompts = [
            "# Write a function to calculate fibonacci numbers\ndef fibonacci(n):",
            "# Two Sum problem\ndef two_sum(nums, target):",
            "# Check if string is palindrome\ndef is_palindrome(s):"
        ]
        
        print(f"\n🧪 TESTING {model_key.upper()}")
        print("=" * 50)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt.split('#')[1].strip()}")
            print(f"Prompt: {prompt}")
            
            try:
                start_time = time.time()
                outputs = generator(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                generation_time = time.time() - start_time
                
                generated_code = outputs[0]['generated_text']
                
                print(f"Generated ({generation_time:.2f}s):")
                print("-" * 30)
                print(generated_code)
                print("-" * 30)
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
        
        print(f"\n✅ Testing complete for {model_key}")
    
    def list_downloaded_models(self) -> List[str]:
        """List models that have been downloaded"""
        downloaded = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                downloaded.append(model_dir.name)
        
        return downloaded
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get information about a downloaded model"""
        model_dir = self.models_dir / model_key
        metadata_file = model_dir / "metadata.json"
        
        if not metadata_file.exists():
            return {"error": "Model not found or metadata missing"}
        
        with open(metadata_file, "r") as f:
            return json.load(f)

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="PyPilot Model Manager")
    parser.add_argument("--model", "-m", default="qwen2.5-coder", 
                       help="Model to download (default: qwen2.5-coder)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available models")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Test downloaded model")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force re-download even if exists")
    parser.add_argument("--models-dir", default="models",
                       help="Directory to store models (default: models)")
    parser.add_argument("--downloaded", "-d", action="store_true",
                       help="List downloaded models")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ModelManager(args.models_dir)
    
    if args.list:
        manager.list_available_models()
        return
    
    if args.downloaded:
        downloaded = manager.list_downloaded_models()
        print(f"\n📂 DOWNLOADED MODELS ({len(downloaded)})")
        print("=" * 40)
        for model_key in downloaded:
            info = manager.get_model_info(model_key)
            config = info.get("config", {})
            print(f"✅ {model_key}")
            print(f"   Size: {config.get('size', 'unknown')}")
            print(f"   Type: {config.get('type', 'unknown')}")
            print(f"   Path: {manager.models_dir / model_key}")
        return
    
    if args.test:
        manager.test_model(args.model)
        return
    
    # Default action: download model
    print(f"\n🚀 PYPILOT MODEL MANAGER")
    print("=" * 40)
    
    try:
        model_path = manager.download_model(args.model, force=args.force)
        print(f"\n✅ SUCCESS!")
        print(f"📁 Model saved to: {model_path}")
        print(f"\n💡 Next steps:")
        print(f"   Test it: python model.py --test --model {args.model}")
        print(f"   Use in code: from model import ModelManager")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()