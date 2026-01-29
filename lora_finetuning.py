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


MODEL_ID = "Qwen/Qwen2.5-Coder-7B"


def format_example(example: dict) -> dict:
    """
    Format a single example into instruction-following format.

    The LeetCode dataset has:
      - prompt: problem description + starter code
      - completion: the solution code
    """
    instruction = example["prompt"]
    response = example["completion"]

    # Qwen chat format
    text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

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
        "torch_dtype": torch.bfloat16,
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


def create_lora_config() -> LoraConfig:
    """Create LoRA configuration for Qwen2.5-Coder."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # LoRA rank
        lora_alpha=32,                 # LoRA alpha (scaling factor)
        lora_dropout=0.05,             # Dropout for LoRA layers
        target_modules=[               # Qwen2.5 attention modules
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

    # Load model and tokenizer
    print(f"Loading model: {MODEL_ID}...")
    model, tokenizer = load_model_and_tokenizer(
        MODEL_ID,
        use_4bit=args.use_4bit,
        use_flash_attention=args.use_flash_attention,
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
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
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        "report_to": "none",  # Set to "wandb" if you want W&B logging
        "push_to_hub": False,
        "max_grad_norm": 0.3,
        "dataloader_num_workers": 4,
        # Completion-only loss: only compute loss on assistant response
        "completion_only_loss": True,
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
        processing_class=tokenizer,  # This is the tokenizer parameter name
        # Note: dataset_text_field, max_seq_length, and packing are not in the signature
        # The dataset already has "text" field from format_example(), so it should work
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
    parser = argparse.ArgumentParser(description="LoRA Finetuning for Qwen2.5-Coder-7B")

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
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
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

    args = parser.parse_args()
    main(args)
