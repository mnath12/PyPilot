"""
Script to show finetuned LoRA model outputs on sample test data points.
Displays problem name, prompt, generated code, and test results.

Usage:
    python show_model_outputs.py --model_path outputs/qwen-lora/final --num_samples 3
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
import argparse
from pathlib import Path
import textwrap

from code_execution import check_compilation, extract_code_from_completion, run_tests

# Qwen chat tokens (match training format)
CHAT_USER = "<|im_start|>user\n"
CHAT_ASSISTANT = "<|im_start|>assistant\n"
CHAT_END = "<|im_end|>"


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
        user_content = prelude + "\n\n" + q
    else:
        user_content = q

    user_content = user_content.strip()
    return f"{CHAT_USER}{user_content}{CHAT_END}\n{CHAT_ASSISTANT}"


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
    
    print(f"Loading base model: {base_model_id}...")
    
    # Quantization config for memory efficiency
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
    print(f"Loading LoRA adapters from: {lora_model_path}...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # Merge adapters for faster inference
    print("Merging LoRA adapters...")
    model = model.merge_and_unload()
    
    print("Model loaded successfully!\n")
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

    # Build list of stop token IDs (EOS + Qwen special tokens)
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # Add Qwen-specific stop tokens
    for stop_token in ["<|im_end|>", "<|file_sep|>", "<|fim_prefix|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(stop_token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop_token_ids.append(token_id)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if stop_token_ids:
        gen_kwargs["eos_token_id"] = stop_token_ids
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text


def format_code_block(code: str, max_width: int = 80) -> str:
    """Format code for display with line wrapping."""
    lines = code.split('\n')
    formatted_lines = []
    for line in lines:
        if len(line) > max_width:
            # Try to wrap at spaces
            wrapped = textwrap.wrap(line, width=max_width, break_long_words=False)
            formatted_lines.extend(wrapped)
        else:
            formatted_lines.append(line)
    return '\n'.join(formatted_lines)


def display_sample(model, tokenizer, row: dict, sample_num: int, include_prelude: bool = True):
    """
    Display model output for a single sample.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        row: Dataset row dictionary
        sample_num: Sample number for display
        include_prelude: Whether to include prelude in prompt
    """
    print("=" * 80)
    print(f"SAMPLE {sample_num}")
    print("=" * 80)
    
    # Get problem information
    task_id = row.get("task_id", "N/A")
    question_id = row.get("question_id", "N/A")
    difficulty = row.get("difficulty", "N/A")
    tags = row.get("tags", [])
    problem_description = row.get("problem_description", "")
    entry_point = row.get("entry_point", "candidate")
    
    # Display problem info
    print(f"\n📋 Problem Information:")
    print(f"   Task ID: {task_id}")
    print(f"   Question ID: {question_id}")
    print(f"   Difficulty: {difficulty}")
    if tags:
        print(f"   Tags: {', '.join(tags)}")
    print(f"   Entry Point: {entry_point}")
    
    # Display problem description (truncated if too long)
    if problem_description:
        desc_lines = problem_description.split('\n')
        print(f"\n📝 Problem Description (first 500 chars):")
        desc_preview = '\n'.join(desc_lines[:10])  # First 10 lines
        if len(problem_description) > 500:
            desc_preview = desc_preview[:500] + "..."
        print(f"   {textwrap.indent(desc_preview, '   ')}")
    
    # Build prompt
    prompt = build_model_prompt(row, include_prelude=include_prelude)
    starter_code = row.get("starter_code", "")
    test_code = row.get("test", "")
    
    # Display prompt (truncated)
    print(f"\n💬 Prompt (first 800 chars):")
    prompt_preview = prompt[:800]
    if len(prompt) > 800:
        prompt_preview += "..."
    print(f"   {textwrap.indent(prompt_preview, '   ')}")
    
    # Generate code
    print(f"\n🤖 Generating code...")
    try:
        generated_completion = generate_code(model, tokenizer, prompt)
    except Exception as e:
        print(f"   ❌ Error generating code: {e}")
        return
    
    # Display raw generation
    print(f"\n📤 Raw Model Output (first 1000 chars):")
    raw_preview = generated_completion[:1000]
    if len(generated_completion) > 1000:
        raw_preview += "..."
    print(f"   {textwrap.indent(raw_preview, '   ')}")
    
    # Extract code
    extracted_code = extract_code_from_completion(generated_completion, starter_code)
    
    # Display extracted code
    print(f"\n💻 Extracted Code:")
    print("   " + "─" * 76)
    code_display = textwrap.indent(extracted_code, '   ')
    print(code_display)
    print("   " + "─" * 76)
    
    # Check compilation
    print(f"\n🔍 Compilation Check:")
    compiles, compile_error = check_compilation(extracted_code)
    if compiles:
        print("   ✅ Code compiles successfully!")
    else:
        print(f"   ❌ Compilation failed: {compile_error}")
    
    # Run tests
    if compiles:
        print(f"\n🧪 Running Tests:")
        tests_passed, test_error = run_tests(extracted_code, test_code, entry_point=entry_point, timeout=10)
        if tests_passed:
            print("   ✅ All tests passed!")
        else:
            print(f"   ❌ Tests failed: {test_error if test_error else 'Unknown error'}")
    else:
        print(f"\n🧪 Tests: Skipped (code doesn't compile)")
    
    print("\n")


def main():
    """Main function to display model outputs."""
    parser = argparse.ArgumentParser(description="Show finetuned LoRA model outputs on sample test data")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the finetuned LoRA model directory (e.g., outputs/qwen-lora/final)")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-Coder-7B",
                        help="Base model identifier (should match training)")
    parser.add_argument("--data_dir", type=str, default="data/leetcode",
                        help="Path to the LeetCode dataset")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of test samples to display")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in test dataset")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (default: True)")
    parser.add_argument("--no_4bit", action="store_false", dest="use_4bit",
                        help="Disable 4-bit quantization")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (only used if sampling is enabled)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_lora_model(args.base_model_id, args.model_path, use_4bit=args.use_4bit)
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = load_from_disk(args.data_dir)
    test_split = dataset["test"]
    print(f"Test set size: {len(test_split)}\n")
    
    # Display samples
    end_idx = min(args.start_idx + args.num_samples, len(test_split))
    print(f"Displaying samples {args.start_idx} to {end_idx-1}:\n")
    
    for i in range(args.start_idx, end_idx):
        if i >= len(test_split):
            print(f"Sample {i} is out of range (dataset has {len(test_split)} samples)")
            break
        
        row = test_split[i]
        display_sample(model, tokenizer, row, sample_num=i+1, include_prelude=True)
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

