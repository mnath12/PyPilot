# Diagnosis: Low Test Pass Rate (0.88%)

## Summary

**Observed**: Compilation rate ~98%, test pass rate &lt;1% (2/228).

**Root cause**: The code sent to the test runner was **only the model's extracted solution**, without the dataset **prompt**. In this dataset, the prompt contains the imports and context required for the tests (e.g. `from typing import ...`). Without it, execution fails with `NameError` (e.g. `List`/`Dict` not defined) or other missing-context errors even when the solution logic is correct.

## Evidence

- **Ground-truth test** (`testing_script.py`) builds the runnable file as:
  `prompt_imports + completion + test_code + check(entry_point)`  
  i.e. the dataset **prompt** is prepended to the solution.
- **Evaluation** was using `full_solution = extracted_code` only (no prompt), so the executed file was missing that context.
- Existing **diagnosis_and_recommendations.md** already notes that missing typing imports cause many failures; the prompt is the intended source of those imports.

## Fixes Applied

1. **Include prompt in the solution used for tests**  
   In `lora_finetuning_test.py`, the runnable code is now:
   - `full_solution = (prompt + "\n\n" + extracted_code)` when `prompt` is non-empty,
   - so the same imports/context as in the ground-truth setup are present when running tests.

2. **Diagnose failures**  
   - `run_tests` in `code_execution.py` now returns up to 1500 characters of error output for better diagnostics.
   - New CLI flag: `--diagnose N`  
     Logs the first N test failure reasons (task_id + error snippet) after evaluation.  
   Example:  
   `python lora_finetuning_test.py --model_path outputs/qwen-lora/final --data_dir data/leetcode --diagnose 10`

3. **Failure breakdown**  
   Evaluation logs a count of failure types (AssertionError, ModuleNotFoundError, NameError, TypeError, etc.) and a hint to install missing deps when ModuleNotFoundError appears.

4. **Dependency**  
   Added `sortedcontainers` to `requirements.txt` for problems that use it (e.g. maximum-value-sum-by-placing-three-rooks).

## What to do next

1. **Re-run evaluation** (no extra args):  
   `python lora_finetuning_test.py --model_path outputs/qwen-lora/final --data_dir data/leetcode`  
   Test pass rate should improve once the runner uses prompt + solution.

2. **Install deps**  
   `pip install sortedcontainers` (or `pip install -r requirements.txt`) so problems requiring it no longer fail with ModuleNotFoundError.

3. **If pass rate is still low**, run with diagnosis to see actual errors:  
   `python lora_finetuning_test.py ... --diagnose 15`  
   Check the **failure breakdown** and sample errors:
   - **AssertionError** → wrong answer; improve model (larger base, more data, more training).
   - **NameError/TypeError** in generated code → bugs in model output.
   - **ModuleNotFoundError** → install the missing package and add to requirements.txt.
   - Timeouts → increase `timeout` or optimize solution.

4. **Optional**: Compilation check can be run on the same `full_solution` (prompt + extracted_code) so "compiles" reflects what is actually executed; currently compilation is still checked on that combined code.

## Follow-up: Diagnose output (March 2026)

From `--diagnose 15`, failures fall into: **AssertionError** (wrong answer), **NameError/TypeError** (bug in generated code), **ModuleNotFoundError** (e.g. `sortedcontainers`). The pipeline now adds `sortedcontainers` to requirements and prints a failure breakdown. After `pip install sortedcontainers`, the two ModuleNotFoundError failures for that package are resolved; most remaining failures are model correctness (try larger model or more training).

## Notebook analysis (code + logs)

**Logs in notebook**: LoRA eval shows **~3% test pass rate** (7/228), **~99.5% compile rate**; base model eval shows **0% pass**. So code compiles but tests fail.

**Cause (same as script)**: The notebook was sending **only `extracted_code`** to `run_tests`, not **prompt + extracted_code**. The dataset `"prompt"` field holds the imports/context (see `testing_script.py`: `full_code = prompt_imports + completion + test_code + check(entry_point)`). Without that prelude, execution hits `NameError` (e.g. `List`/`Dict` not defined) even when the solution logic is correct.

**Fix applied in notebook**:
- **LoRA evaluation cell**: `prelude = (row.get("prompt") or "").strip()` and `full_solution = (prelude + "\n\n" + extracted_code).strip() if prelude else extracted_code` before `check_compilation` and `run_tests`.
- **Base model evaluation cell**: Same logic; `run_tests(full_solution, ...)` instead of `run_tests(extracted, ...)`.

Re-run the evaluation cells in the notebook to see the improved pass rate (and install `sortedcontainers` if you hit ModuleNotFoundError).
