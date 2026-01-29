"""
Evaluation loop for testing Qwen2.5-Coder-14B on LeetCodeDataset test data.
Logs compilation rate and test pass rate.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
import logging
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


def load_model_and_tokenizer(model_id: str = "Qwen/Qwen2.5-Coder-14B"):
    """
    Load the model and tokenizer.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
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


def evaluate_on_dataset(model, tokenizer, dataset, max_samples: int = None, include_prelude: bool = False) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: Test dataset
        max_samples: Maximum number of samples to evaluate (None for all)
        include_prelude: Whether to include prelude in the prompt (default: False)
        
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
            generated_completion = generate_code(model, tokenizer, prompt)
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
    # Load model
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-Coder-7B")
    
    # Load dataset
    logger.info("Loading dataset from disk...")
    dataset = load_from_disk("data/leetcode")
    logger.info(f"Dataset loaded. Test set size: {len(dataset['test'])}")
    
    # Evaluate
    metrics = evaluate_on_dataset(model, tokenizer, dataset, max_samples=None, include_prelude=True)
    
    # Log final results
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples evaluated: {metrics['total_evaluated']}")
    logger.info(f"Compilation rate: {metrics['compile_rate']:.2f}% ({metrics['compile_count']}/{metrics['total_evaluated']})")
    logger.info(f"Test pass rate: {metrics['test_pass_rate']:.2f}% ({metrics['test_pass_count']}/{metrics['total_evaluated']})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

