# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyPilot is an LLM-based coding assistant that generates Python programs and solves LeetCode-style problems from natural language prompts. The core engine uses Qwen2.5-Coder-7B as the base model with LoRA fine-tuning for parameter efficiency.

**Development Roadmap:**
1. Baseline evaluation (no finetuning) — establish reference metrics
2. Supervised finetuning (SFT/PEFT) — improve on target task distribution
3. RL stage (PPO-style) — optimize reward tied to compilation and test success

## Commands

### Setup
```bash
pip install -r requirements.txt
python check_cuda.py  # Verify GPU/CUDA availability
```

### Data Preparation
```bash
# Download LeetCodeDataset from HuggingFace
python prepare_leetcode_dataset.py --out data/leetcode --train_frac 1.0
```

### Training
```bash
# LoRA fine-tuning
python lora_finetuning.py --data_dir data/leetcode --output_dir outputs/qwen-lora --epochs 3 --batch_size 2

# Resume from checkpoint
python lora_finetuning.py --resume_from_checkpoint outputs/qwen-lora/checkpoint-400 --data_dir data/leetcode --output_dir outputs/qwen-lora
```

### Evaluation
```bash
# Baseline model (no fine-tuning)
python test_model.py

# Finetuned model
python lora_finetuning_test.py --model_path outputs/qwen-lora/final --data_dir data/leetcode

# Interactive sample outputs
python show_model_outputs.py --model_path outputs/qwen-lora/final --num_samples 3
```

## Architecture

### Training Pipeline
1. LeetCodeDataset is formatted to Qwen chat format: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>`
2. Base model loaded with 4-bit quantization (BitsAndBytes)
3. LoRA adapters applied (rank=4, alpha=4, targets all attention/MLP projections)
4. SFTTrainer with completion-only loss, cosine scheduler, paged_adamw_8bit optimizer

### Inference/Evaluation Pipeline
1. Build prompt with Qwen chat format
2. Generate with deterministic sampling (temperature=0)
3. Extract code: strip chat markers, extract from markdown blocks, add typing imports
4. Check compilation via `ast.parse()`
5. Execute tests in subprocess with timeout

### Key Modules
- **lora_finetuning.py**: Main LoRA training script with SFTTrainer
- **lora_finetuning_test.py**: Evaluate finetuned models (compile rate, test pass rate)
- **test_model.py**: Evaluate baseline model without fine-tuning
- **code_execution.py**: Code extraction, compilation checking, test execution with subprocess isolation
- **prepare_leetcode_dataset.py**: Download and cache LeetCodeDataset from HuggingFace

### Dataset Schema (LeetCodeDataset)
Key fields: `task_id`, `difficulty`, `entry_point`, `problem_description`, `prompt`, `starter_code`, `test`, `completion`

## Code Style Guidelines

From context.md:
- Keep changes small and localized; avoid broad refactors unless necessary
- Preserve existing APIs and entry points
- Add docstrings + type hints in new code
- Prefer explicit, readable Python over clever tricks
- For training code: include fixed random seeds, clear config surface, logging hooks

## Known Issues

The code extraction in `code_execution.py` strips typing imports. Generated code using `List[int]`, `Dict[str, int]` etc. will fail with `NameError`. The extraction logic attempts to auto-add imports but may need improvement.
