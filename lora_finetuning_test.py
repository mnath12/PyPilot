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
from collections import Counter
from pathlib import Path
from typing import Dict

from code_execution import check_compilation, extract_code_from_completion, run_tests


def _categorize_failure(error_msg: str) -> str:
    """Bucket test failure for diagnostics. Returns a short label."""
    if not error_msg:
        return "Unknown"
    if "ModuleNotFoundError" in error_msg:
        return "ModuleNotFoundError"
    if "AssertionError" in error_msg:
        return "AssertionError"
    if "NameError" in error_msg:
        return "NameError"
    if "TypeError" in error_msg:
        return "TypeError"
    if "Timeout" in error_msg or "TimeoutExpired" in error_msg:
        return "Timeout"
    if "SyntaxError" in error_msg:
        return "SyntaxError"
    return "Other"


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Qwen chat tokens (match training format)
CHAT_USER = "<|im_start|>user\n"
CHAT_ASSISTANT = "<|im_start|>assistant\n"
CHAT_END = "<|im_end|>"


def build_model_prompt(row: dict, include_prelude: bool = False) -> str:
    """
    Build the model prompt from dataset row.

    Uses query-only and same cleaning as training (notebook behavior).
    include_prelude is kept for API compatibility but not used (query-only).
    """
    user_content = (row.get("query") or "").strip()
    user_content = user_content.replace("(use the provided format with backticks)", "")
    user_content = user_content.replace("and enclose your code within delimiters.", "")
    user_content = user_content.rstrip()
    user_content += "\n\nRespond with only the Python code. No explanations, no markdown."
    return f"{CHAT_USER}{user_content}{CHAT_END}\n{CHAT_ASSISTANT}"


def clean_code(text: str) -> str:
    """
    Keep only the first class Solution block and remove trailing junk.
    Matches notebook post-processing for LeetCode-style outputs.
    """
    start = text.find("class Solution:")
    if start == -1:
        return text.strip()

    text = text[start:]
    second = text.find("\nclass Solution:", 10)
    if second != -1:
        text = text[:second]

    lines = text.splitlines()
    if lines:
        last = lines[-1]
        if (
            not last.startswith(" ")
            and not last.startswith("\t")
            and "class" not in last
            and "def" not in last
            and "=" not in last
        ):
            lines = lines[:-1]
    return "\n".join(lines).strip()


def load_lora_model(
    base_model_id: str,
    lora_model_path: str,
    use_4bit: bool = True,
    merge_adapters: bool = True,
):
    """
    Load the base model and apply LoRA adapters.

    Args:
        base_model_id: HuggingFace model identifier for the base model
        lora_model_path: Path to the finetuned LoRA adapters
        use_4bit: Whether to use 4-bit quantization
        merge_adapters: If True, merge LoRA into base for faster inference

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
    model.config.pad_token_id = tokenizer.eos_token_id

    if merge_adapters:
        logger.info("Merging LoRA adapters for faster inference...")
        model = model.merge_and_unload()

    logger.info("Model loaded successfully")
    return model, tokenizer


def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> str:
    """
    Generate code from a prompt with FIM/repo tokens banned and hard cut on stop markers.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    eos_ids = []
    for tok in ["<|im_end|>", "<|endoftext|>", "<|file_sep|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)

    banned = [
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
    ]
    bad_words_ids = []
    for tok in banned:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            bad_words_ids.append(ids)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": eos_ids,
        "bad_words_ids": bad_words_ids if bad_words_ids else None,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Hard cut on stop markers
    for marker in [
        "<|im_end|>",
        "<|endoftext|>",
        "<|repo_name|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
    ]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    return text.strip()


def evaluate_on_dataset(
    model,
    tokenizer,
    dataset,
    max_samples: int = None,
    include_prelude: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    diagnose_failures: int = 0,
) -> Dict[str, float]:
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
        diagnose_failures: If > 0, log this many test failure reasons (task_id + error snippet)
        
    Returns:
        Dictionary with metrics: compile_rate, test_pass_rate, total_samples
    """
    test_split = dataset["test"]
    total_samples = len(test_split) if max_samples is None else min(max_samples, len(test_split))
    
    logger.info(f"Evaluating on {total_samples} samples from test set")
    
    compile_count = 0
    test_pass_count = 0
    total_evaluated = 0
    failure_log: list = []  # (task_id, error_msg) for diagnose_failures
    failure_categories: list = []  # error type per failed test for breakdown

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
        
        # Generate code and clean to first class Solution block (notebook behavior)
        try:
            raw_completion = generate_code(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
            )
            generated_completion = clean_code(raw_completion)
        except Exception as e:
            logger.error(f"Error generating code for {task_id}: {e}")
            continue

        # Extract code from completion
        extracted_code = extract_code_from_completion(generated_completion, starter_code)

        # Build full solution: dataset "prompt" contains required imports/context for tests (see testing_script.py).
        # Without it, tests fail with NameError (e.g. List, Dict) or missing context.
        prelude = (row.get("prompt") or "").strip()
        full_solution = (prelude + "\n\n" + extracted_code).strip() if prelude else extracted_code

        compiles, compile_error = check_compilation(full_solution)

        if compiles:
            compile_count += 1
        else:
            logger.debug(f"Compilation failed for {task_id}: {compile_error}")
        
        # Run tests (only if code compiles)
        tests_passed = False
        if compiles:
            try:
                tests_passed, test_error = run_tests(full_solution, test_code, entry_point=entry_point, timeout=10)
                if tests_passed:
                    test_pass_count += 1
                else:
                    logger.debug(f"Tests failed for {task_id}: {test_error}")
                    failure_categories.append(_categorize_failure(test_error or ""))
                    if diagnose_failures and len(failure_log) < diagnose_failures:
                        failure_log.append((task_id, (test_error or "")[:800]))
            except Exception as e:
                logger.debug(f"Error running tests for {task_id}: {e}")
                failure_categories.append(_categorize_failure(str(e)))
        
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
    
    # Log first N failure reasons when diagnosing
    if failure_log:
        logger.info("--- Sample test failure reasons (diagnose) ---")
        for i, (tid, err) in enumerate(failure_log, 1):
            logger.info(f"  [{i}] {tid}: {err}")
        logger.info("--- end diagnose ---")

    # Failure breakdown (all failed tests)
    if failure_categories:
        breakdown = Counter(failure_categories)
        logger.info(
            "Failure breakdown: " + ", ".join(f"{k}={v}" for k, v in breakdown.most_common())
        )
        if "ModuleNotFoundError" in breakdown:
            logger.info(
                "  -> Install missing deps (e.g. pip install sortedcontainers) to fix ModuleNotFoundError."
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
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
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
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (only used if sampling is enabled)")
    parser.add_argument("--no_merge_adapters", action="store_true",
                        help="Do not merge LoRA adapters (keep separate for debugging)")
    parser.add_argument("--diagnose", type=int, default=0, metavar="N",
                        help="Log first N test failure reasons (task_id + error) to diagnose low pass rate")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_lora_model(
        args.base_model_id,
        args.model_path,
        use_4bit=args.use_4bit,
        merge_adapters=not args.no_merge_adapters,
    )
    
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
        include_prelude=True,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        diagnose_failures=args.diagnose,
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

