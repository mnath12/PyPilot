"""
LoRA Finetuning script for Qwen2.5-Coder-7B on LeetCode Dataset.

Usage:
    python lora_finetuning.py --data_dir data/leetcode --output_dir outputs/qwen-lora

Requirements:
    pip install torch transformers datasets peft accelerate bitsandbytes trl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# Default base model (use --model_id to override, e.g. Qwen/Qwen2.5-Coder-7B-Instruct)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Qwen chat tokens (match training/inference format)
CHAT_USER = "<|im_start|>user\n"
CHAT_ASSISTANT = "<|im_start|>assistant\n"
CHAT_END = "<|im_end|>"


def format_example(example: dict) -> dict:
    """
    Format a single example into instruction-following format.

    Uses query-only (no prompt/prelude) and instructs the model to respond
    with only Python code, matching the notebook training setup.
    """
    instruction = (example.get("query") or "").strip()
    instruction = instruction.replace("(use the provided format with backticks)", "")
    instruction = instruction.replace("and enclose your code within delimiters.", "")
    instruction = instruction.rstrip()
    instruction += "\n\nRespond with only the Python code. No explanations, no markdown."

    response = example["completion"]
    # Qwen chat format (no system prompt)
    text = f"{CHAT_USER}{instruction}{CHAT_END}\n{CHAT_ASSISTANT}{response}{CHAT_END}"
    return {"text": text}


def load_model_and_tokenizer(
    model_id: str,
    use_4bit: bool = True,
    use_flash_attention: bool = False,
):
    """Load Qwen2.5-Coder with optional 4-bit quantization."""

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
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )



    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    # Prepare model for k-bit training if using quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def check_bf16_support() -> Tuple[bool, str]:
    """
    Check if the current setup supports bfloat16.
    
    Returns:
        Tuple of (supports_bf16: bool, info_message: str)
    """
    info_lines = []
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    
    info_lines.append(f"CUDA available: {torch.cuda.is_available()}")
    info_lines.append(f"CUDA version: {torch.version.cuda}")
    info_lines.append(f"PyTorch version: {torch.__version__}")
    
    # Check GPU info
    try:
        device_count = torch.cuda.device_count()
        info_lines.append(f"GPU count: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            info_lines.append(f"GPU {i}: {gpu_name} (Compute Capability: {device_capability[0]}.{device_capability[1]})")
            
            # Check if compute capability supports bf16 (Ampere 8.0+ or Ada Lovelace 8.9+)
            if device_capability[0] >= 8:
                info_lines.append(f"  -> GPU {i} architecture supports bf16 (Ampere/Ada Lovelace)")
            else:
                info_lines.append(f"  -> GPU {i} architecture may not support bf16 (pre-Ampere)")
        
        # Try to create a bf16 tensor to test actual support
        try:
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda:0")
            info_lines.append("bf16 tensor creation test: SUCCESS")
            supports_bf16 = True
        except Exception as e:
            info_lines.append(f"bf16 tensor creation test: FAILED - {e}")
            supports_bf16 = False
            
    except Exception as e:
        info_lines.append(f"Error checking GPU: {e}")
        supports_bf16 = False
    
    info_message = "\n".join(info_lines)
    return supports_bf16, info_message


def create_lora_config(
    r: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> LoraConfig:
    """Create LoRA configuration for Qwen2.5-Coder.

    Matches the notebook implementation:
    - rank r=4, alpha=16, dropout=0.1
    - Applied to attention and MLP projection layers
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )


def main(args):
    model_id = args.model_id or DEFAULT_MODEL_ID
    print(f"Loading dataset from {args.data_dir}...")
    dataset = load_from_disk(args.data_dir)

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")

    # Format dataset
    print("Formatting dataset...")
    train_dataset = dataset["train"].map(
        format_example,
        remove_columns=dataset["train"].column_names,
        desc="Formatting train",
    )
    eval_dataset = dataset["test"].map(
        format_example,
        remove_columns=dataset["test"].column_names,
        desc="Formatting eval",
    )

    # Optionally limit dataset size for faster iteration
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    print(f"Training on {len(train_dataset)} samples, evaluating on {len(eval_dataset)} samples")

    # Optional: print token length stats and truncation warning (notebook behavior)
    if getattr(args, "print_length_stats", True) and len(train_dataset) > 0:
        import numpy as np
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        lengths = []
        for ex in train_dataset.select(range(min(500, len(train_dataset)))):
            ids = _tok(ex["text"], truncation=False, add_special_tokens=False)["input_ids"]
            lengths.append(len(ids))
        lengths = np.array(lengths)
        max_len = args.max_seq_length
        truncated = (lengths > max_len).sum()
        print(f"Token length stats (sample): min={lengths.min()}, median={int(np.median(lengths))}, p95={int(np.percentile(lengths, 95))}, max={lengths.max()}")
        print(f"Truncated (>{max_len}): {truncated}/{len(lengths)} ({100 * truncated / len(lengths):.1f}%)")
        if truncated / len(lengths) > 0.10:
            print(f"  Consider increasing --max_seq_length (e.g. to {int(np.percentile(lengths, 95))})")

    # Load model and tokenizer
    print(f"Loading model: {model_id}...")
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        use_4bit=args.use_4bit,
        use_flash_attention=args.use_flash_attention,
    )

    tokenizer.model_max_length = args.max_seq_length


    # Apply LoRA
    print("Applying LoRA adapters...")
    lora_config = create_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    
    # Verify that base weights are frozen (PEFT should do this automatically)
    # Count trainable vs total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    model.print_trainable_parameters()

    # Check bf16 support
    print("\n" + "="*60)
    print("Checking bf16 support...")
    print("="*60)
    supports_bf16, bf16_info = check_bf16_support()
    print(bf16_info)
    print("="*60 + "\n")
    
    # Determine mixed precision settings
    if supports_bf16:
        print("Using bfloat16 mixed precision training")
        use_bf16 = True
        use_fp16 = False
    elif torch.cuda.is_available():
        print("bf16 not supported, falling back to float16 mixed precision")
        use_bf16 = False
        use_fp16 = True
    else:
        print("No GPU available, using full precision (fp32)")
        use_bf16 = False
        use_fp16 = False

    # SFTConfig with completion-only loss (replaces TrainingArguments + DataCollatorForCompletionOnlyLM)
    output_dir = Path(args.output_dir)
    
    # Build config kwargs
    config_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        "report_to": "none",
        "push_to_hub": False,
        "max_grad_norm": 1.0,
        "dataloader_num_workers": 0,  # IMPORTANT on Windows
        "completion_only_loss": True,
        "dataset_text_field": "text",
        "max_length": args.max_seq_length,
    }

    # Add mixed precision settings
    if use_bf16:
        config_kwargs["bf16"] = True
        config_kwargs["fp16"] = False
    elif use_fp16:
        config_kwargs["bf16"] = False
        config_kwargs["fp16"] = True
    else:
        config_kwargs["bf16"] = False
        config_kwargs["fp16"] = False
    
    # Add gradient checkpointing kwargs if enabled
    if args.gradient_checkpointing:
        config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    
    sft_config = SFTConfig(**config_kwargs)

    # Initialize trainer
    # Based on actual SFTTrainer signature from TRL
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    final_model_path = output_dir / "final"
    print(f"Saving final model to {final_model_path}...")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Finetuning for Qwen2.5-Coder")

    # Model arguments
    parser.add_argument("--model_id", type=str, default=None,
                        help=f"Base model ID (default: {DEFAULT_MODEL_ID}). Use Qwen/Qwen2.5-Coder-7B-Instruct for chat model.")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/leetcode",
                        help="Path to the LeetCode dataset (saved with datasets.save_to_disk)")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen-lora",
                        help="Directory to save checkpoints and final model")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples for debugging")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Limit eval samples for faster evaluation")

    # Model arguments
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (default: True)")
    parser.add_argument("--no_4bit", action="store_false", dest="use_4bit",
                        help="Disable 4-bit quantization")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use Flash Attention 2 (requires compatible GPU)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4 for LoRA fine-tuning)")
    parser.add_argument("--max_seq_length", type=int, default=3072,
                        help="Maximum sequence length (default 3072 to reduce truncation)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--no_gradient_checkpointing", action="store_false",
                        dest="gradient_checkpointing")

    # Checkpoint arguments
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # LoRA arguments (notebook defaults)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--no_print_length_stats", action="store_true",
                        help="Skip printing token length / truncation stats")

    args = parser.parse_args()
    args.print_length_stats = not args.no_print_length_stats
    main(args)
