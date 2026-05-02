"""Quick sanity check: do ground-truth completions pass the test harness?"""
from datasets import load_from_disk

dataset = load_from_disk("data/leetcode")
test_split = dataset["test"]

passed = 0
failed = 0
errors = []

for i in range(min(20, len(test_split))):
    row = test_split[i]
    prompt_imports = row["prompt"]       # the imports
    completion = row["completion"]       # ground truth solution
    test_code = row["test"]             # def check(candidate): ...
    entry_point = row["entry_point"]    # e.g. "Solution().shortestDistanceAfterQueries"

    full_code = f"""{prompt_imports}

{completion}

{test_code}

check({entry_point})
"""
    try:
        exec(full_code, {})
        passed += 1
    except Exception as e:
        failed += 1
        errors.append((i, row["task_id"], type(e).__name__, str(e)[:200]))

print(f"Passed: {passed}/{passed + failed}")
print(f"Failed: {failed}/{passed + failed}")
for idx, task_id, err_type, err_msg in errors:
    print(f"  [{idx}] {task_id}: {err_type}: {err_msg}")