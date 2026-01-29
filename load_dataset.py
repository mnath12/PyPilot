from datasets import load_from_disk
import textwrap
ds = load_from_disk("data/leetcode")
print(ds)

row = ds["test"][0]  # change index as you like

print("=== Basic fields ===")
print("task_id:", row["task_id"])
print("question_id:", row["question_id"])
print("difficulty:", row["difficulty"])
print("estimated_date:", row["estimated_date"])
print("entry_point:", row["entry_point"])
print("tags:", row["tags"])

print("=== Description ===")
print(textwrap.shorten(row["problem_description"].replace("\n", " "), width=1200, placeholder=" ..."))

print("\n=== Prompt (first 1200 chars) ===")
print(textwrap.shorten(row["prompt"].replace("\n", " "), width=1200, placeholder=" ..."))

print("\n=== Starter code (first 800 chars) ===")
starter = row.get("starter_code") or ""
print(starter[:800] + ("..." if len(starter) > 800 else ""))

print("\n=== Test (first 800 chars) ===")
test = row.get("test") or ""
print(test[:800] + ("..." if len(test) > 800 else ""))

print("\n=== Completion (first 800 chars) ===")
comp = row.get("completion") or ""
print(comp[:800] + ("..." if len(comp) > 800 else ""))








