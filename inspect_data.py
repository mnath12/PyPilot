"""Inspect training data format to diagnose the model confusion issue."""
from datasets import load_from_disk

ds = load_from_disk("data/leetcode")

print("=" * 80)
print("INSPECTING TRAINING DATA FORMAT")
print("=" * 80)

for i in range(3):
    row = ds["train"][i]
    print(f"\n{'='*80}")
    print(f"SAMPLE {i}")
    print("=" * 80)

    print(f"\n--- task_id ---")
    print(row.get("task_id", "N/A"))

    print(f"\n--- query (first 500 chars) ---")
    query = row.get("query", "")
    print(query[:500] if query else "(empty)")

    print(f"\n--- completion (first 800 chars) ---")
    completion = row.get("completion", "")
    print(completion[:800] if completion else "(empty)")

    print(f"\n--- Does completion contain 'You are an expert'? ---")
    print("You are an expert" in completion)

    print(f"\n--- Does completion contain 'class Solution'? ---")
    print("class Solution" in completion)

print("\n" + "=" * 80)
print("CHECKING FOR 'You are an expert' IN ALL COMPLETIONS")
print("=" * 80)

count = 0
for row in ds["train"]:
    if "You are an expert" in row.get("completion", ""):
        count += 1

print(f"Found 'You are an expert' in {count}/{len(ds['train'])} training completions")

count_query = 0
for row in ds["train"]:
    if "You are an expert" in row.get("query", ""):
        count_query += 1

print(f"Found 'You are an expert' in {count_query}/{len(ds['train'])} training queries")
