# PyPilot

**Suggested GitHub “About” description** (copy into the repository description field, ≤350 characters):

> Fine-tune Qwen2.5-Coder with LoRA on LeetCode-style problems: download data, train with TRL/PEFT, and measure compile rate and hidden-test pass rate on `newfacade/LeetCodeDataset`.

PyPilot is a research codebase for **supervised fine-tuning** of code LLMs on LeetCode-style Python tasks. It formats prompts in Qwen chat style, trains **LoRA** adapters with 4-bit loading where supported, and evaluates generations by **AST compilation** plus **subprocess-isolated** execution of dataset tests.

## Roadmap

1. **Baseline evaluation** — run the base model on the test split and record compile / test-pass metrics.  
2. **SFT / PEFT** — LoRA fine-tuning on the training split (`lora_finetuning.py`).  
3. **RL** (future) — reward shaping from compilation and test success (see notebooks such as `pypilot_rl_train.ipynb` if present in your clone).

## Requirements

- Python 3.10+ recommended  
- **NVIDIA GPU with CUDA** strongly recommended for training and full-model eval  
- Hugging Face account not required for public weights; dataset pulls use `datasets`

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
python check_cuda.py
```

## Data

Download and cache [newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset) to disk (adjust `--train_frac` for quick experiments):

```bash
python prepare_leetcode_dataset.py --out data/leetcode --train_frac 1.0
```

Relevant columns include `query`, `prompt`, `completion`, `starter_code`, `test`, `entry_point`, and `task_id` (exact schema may vary by dataset revision).

## Training (LoRA)

Default base checkpoint in `lora_finetuning.py` is `Qwen/Qwen2.5-Coder-0.5B-Instruct` for lighter runs; use `--model_id` for larger models (for example `Qwen/Qwen2.5-Coder-7B-Instruct`).

```bash
python lora_finetuning.py --data_dir data/leetcode --output_dir outputs/qwen-lora --epochs 3 --batch_size 2
```

Resume from a checkpoint:

```bash
python lora_finetuning.py --resume_from_checkpoint outputs/qwen-lora/checkpoint-400 --data_dir data/leetcode --output_dir outputs/qwen-lora
```

Notable flags: `--max_train_samples`, `--max_eval_samples`, `--lora_r`, `--lora_alpha`, `--no_4bit`, `--use_flash_attention`, `--max_seq_length`, `--gradient_accumulation_steps`, `--learning_rate`.

Training uses **TRL `SFTTrainer`** with completion-only loss, cosine schedule, and (when using 4-bit) **paged AdamW 8-bit**. Checkpoints and a `final` adapter folder are written under `--output_dir`.

## Evaluation

**Fine-tuned model** (base + LoRA from `--model_path`):

```bash
python lora_finetuning_test.py --model_path outputs/qwen-lora/final --data_dir data/leetcode
```

Optional: `--max_samples`, `--base_model_id`, `--max_new_tokens`, `--temperature`, `--diagnose N` for failure buckets.

**Baseline** (script currently loads `Qwen/Qwen2.5-Coder-7B` and a fixed sample cap — inspect `test_model.py` to change model id or `max_samples`):

```bash
python test_model.py
```

**Qualitative inspection**:

```bash
python show_model_outputs.py --model_path outputs/qwen-lora/final --num_samples 3
```

Metrics are driven by `code_execution.py`: extract generated Python, `ast.parse` for syntax, then run dataset tests in a subprocess with a timeout.

## Project layout

| Path | Role |
|------|------|
| `prepare_leetcode_dataset.py` | Load HF dataset and `save_to_disk` |
| `lora_finetuning.py` | LoRA SFT training |
| `lora_finetuning_test.py` | Eval adapters: compile + test pass rates |
| `test_model.py` | Baseline eval (base model only) |
| `show_model_outputs.py` | Pretty-print a few generations |
| `code_execution.py` | Code extraction, compile check, test runner |
| `check_cuda.py` | Quick CUDA sanity check |

## Known limitations

- **Prompt alignment**: Training/eval for LoRA paths use a query-only prompt with a fixed “code only” suffix; `test_model.py` may use a different prompt shape (`include_prelude`). Compare scripts before interpreting baseline vs fine-tuned numbers.  
- **Extracted code**: Post-processing strips or adjusts imports and markdown; generated code that relies on typing names without imports can still fail at runtime with `NameError`. See `code_execution.py` and CLAUDE.md in the repo for context.

## Contributing

Keep changes focused; match existing style and CLI patterns. Internal notes for automation live in `CLAUDE.md`.
