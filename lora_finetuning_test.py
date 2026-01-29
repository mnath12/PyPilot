"""
Test script for evaluating finetuned LoRA model performance on LeetCodeDataset test data.
Logs compilation rate and test pass rate.

Usage:
    python lora_finetuning_test.py --model_path outputs/qwen-lora/final --data_dir data/leetcode
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
from typing import Dict

from code_execution import check_compilation, extract_code_from_completion, run_tests


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_model_prompt(row: dict, include_prelude: bool = False) -> str:
    """
    Build the model prompt from dataset row.
    
    Args:
        row: Dataset row dictionary
        include_prelude: Whether to include the prelude (prompt field) before query
        
    Returns:
        Formatted prompt string
    """
    q = (row.get("query") or "").strip()
    prelude = (row.get("prompt") or "").strip()

    if include_prelude and prelude:
        return prelude + "\n\n" + q + "\n"
    return q + "\n"


def load_lora_model(base_model_id: str, lora_model_path: str, use_4bit: bool = True):
    """
    Load the base model and apply LoRA adapters.
    
    Args:
        base_model_id: HuggingFace model identifier for the base model
        lora_model_path: Path to the finetuned LoRA adapters
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import BitsAndBytesConfig
    
    logger.info(f"Loading base model: {base_model_id}")
    
    # Quantization config for memory efficiency (same as training)
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    
    # Load LoRA adapters
    logger.info(f"Loading LoRA adapters from: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # Merge adapters for faster inference (optional - comment out if you want to keep them separate)
    logger.info("Merging LoRA adapters for faster inference...")
    model = model.merge_and_unload()
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def generate_code(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Generate code from a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated code string
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text


def evaluate_on_dataset(model, tokenizer, dataset, max_samples: int = None, include_prelude: bool = True, max_new_tokens: int = 512, temperature: float = 0.7) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: Test dataset
        max_samples: Maximum number of samples to evaluate (None for all)
        include_prelude: Whether to include prelude in the prompt (default: True, matching training)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for generation
        
    Returns:
        Dictionary with metrics: compile_rate, test_pass_rate, total_samples
    """
    test_split = dataset["test"]
    total_samples = len(test_split) if max_samples is None else min(max_samples, len(test_split))
    
    logger.info(f"Evaluating on {total_samples} samples from test set")
    
    compile_count = 0
    test_pass_count = 0
    total_evaluated = 0
    
    for idx in tqdm(range(total_samples), desc="Evaluating"):
        row = test_split[idx]
        
        # Build prompt using build_model_prompt function
        prompt = build_model_prompt(row, include_prelude=include_prelude)
        starter_code = row.get("starter_code", "")
        test_code = row.get("test", "")
        entry_point = row.get("entry_point", "candidate")
        task_id = row.get("task_id", f"task_{idx}")
        
        if not prompt.strip() or not test_code:
            logger.warning(f"Skipping sample {idx}: missing query/prompt or test code")
            continue
        
        # Generate code
        try:
            generated_completion = generate_code(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        except Exception as e:
            logger.error(f"Error generating code for {task_id}: {e}")
            continue
        
        # Extract code from completion
        extracted_code = extract_code_from_completion(generated_completion, starter_code)
        
        # Check compilation
        compiles, compile_error = check_compilation(extracted_code)
        if compiles:
            compile_count += 1
        else:
            logger.debug(f"Compilation failed for {task_id}: {compile_error}")
        
        # Run tests (only if code compiles)
        tests_passed = False
        if compiles:
            try:
                tests_passed, test_error = run_tests(extracted_code, test_code, entry_point=entry_point, timeout=10)
                if tests_passed:
                    test_pass_count += 1
                else:
                    logger.debug(f"Tests failed for {task_id}: {test_error}")
            except Exception as e:
                logger.debug(f"Error running tests for {task_id}: {e}")
        
        total_evaluated += 1
        
        # Log progress periodically
        if (idx + 1) % 10 == 0:
            current_compile_rate = (compile_count / total_evaluated) * 100
            current_test_rate = (test_pass_count / total_evaluated) * 100
            logger.info(
                f"Progress: {idx + 1}/{total_samples} | "
                f"Compile rate: {current_compile_rate:.2f}% | "
                f"Test pass rate: {current_test_rate:.2f}%"
            )
    
    # Calculate final metrics
    compile_rate = (compile_count / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    test_pass_rate = (test_pass_count / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    
    return {
        "compile_rate": compile_rate,
        "test_pass_rate": test_pass_rate,
        "compile_count": compile_count,
        "test_pass_count": test_pass_count,
        "total_evaluated": total_evaluated
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Test finetuned LoRA model on LeetCodeDataset")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the finetuned LoRA model directory (e.g., outputs/qwen-lora/final)")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-Coder-7B",
                        help="Base model identifier (should match training)")
    parser.add_argument("--data_dir", type=str, default="data/leetcode",
                        help="Path to the LeetCode dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (None for all)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (default: True)")
    parser.add_argument("--no_4bit", action="store_false", dest="use_4bit",
                        help="Disable 4-bit quantization")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_lora_model(args.base_model_id, args.model_path, use_4bit=args.use_4bit)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}...")
    dataset = load_from_disk(args.data_dir)
    logger.info(f"Dataset loaded. Test set size: {len(dataset['test'])}")
    
    # Evaluate
    metrics = evaluate_on_dataset(
        model, 
        tokenizer, 
        dataset, 
        max_samples=args.max_samples,
        include_prelude=True,  # Match training setting
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Log final results
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Base model: {args.base_model_id}")
    logger.info(f"Total samples evaluated: {metrics['total_evaluated']}")
    logger.info(f"Compilation rate: {metrics['compile_rate']:.2f}% ({metrics['compile_count']}/{metrics['total_evaluated']})")
    logger.info(f"Test pass rate: {metrics['test_pass_rate']:.2f}% ({metrics['test_pass_count']}/{metrics['total_evaluated']})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

