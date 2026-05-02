"""
Inspect your LeetCode dataset to diagnose training/eval mismatch.

Usage:
    python inspect_data.py --data_dir data/leetcode --num_samples 5

This script helps you:
1. See what raw fields look like (prompt, query, completion)
2. Compare training format vs evaluation format side-by-side
3. Check if completions contain markdown/explanation (bad for code extraction)
"""

import argparse
from datasets import load_from_disk

# ─── Copy of training format_example ───
CHAT_USER = "<|im_start|>user\n"
CHAT_ASSISTANT = "<|im_start|>assistant\n"
CHAT_END = "<|im_end|>"

def training_format(example: dict) -> str:
    """Exact copy of format_example from lora_finetuning.py"""
    prompt = (example.get("prompt") or "").strip()
    query = (example.get("query") or "").strip()
    if prompt and query:
        instruction = prompt + "\n\n" + query
    elif query:
        instruction = query
    else:
        instruction = prompt
    response = example["completion"]
    text = f"{CHAT_USER}{instruction}{CHAT_END}\n{CHAT_ASSISTANT}{response}{CHAT_END}"
    return text

def eval_format(row: dict, include_prelude: bool = True) -> str:
    """Exact copy of build_model_prompt from lora_finetuning_test.py"""
    q = (row.get("query") or "").strip()
    prelude = (row.get("prompt") or "").strip()
    if include_prelude and prelude:
        user_content = prelude + "\n\n" + q
    else:
        user_content = q
    user_content = user_content.strip()
    return f"{CHAT_USER}{user_content}{CHAT_END}\n{CHAT_ASSISTANT}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    args = parser.parse_args()

    dataset = load_from_disk(args.data_dir)
    split = dataset[args.split]

    print(f"Dataset split: {args.split}, size: {len(split)}")
    print(f"Columns: {split.column_names}")
    print()

    for i in range(min(args.num_samples, len(split))):
        row = split[i]
        print("=" * 80)
        print(f"SAMPLE {i}")
        print("=" * 80)

        # ── Raw fields ──
        print("\n--- RAW FIELDS ---")
        for key in ["task_id", "prompt", "query", "completion", "starter_code", "test", "entry_point"]:
            val = row.get(key, "<MISSING>")
            if val and isinstance(val, str) and len(val) > 300:
                val = val[:300] + "... [TRUNCATED]"
            print(f"  {key}: {repr(val)}")

        # ── Check completion content ──
        completion = row.get("completion", "")
        print("\n--- COMPLETION ANALYSIS ---")
        print(f"  Length: {len(completion)} chars")
        print(f"  Starts with '```': {completion.strip().startswith('```')}")
        print(f"  Contains '```python': {'```python' in completion}")
        print(f"  Contains '```': {'```' in completion}")
        print(f"  Contains 'def ': {'def ' in completion}")
        print(f"  Contains 'class ': {'class ' in completion}")
        # Check if it has natural language explanation
        explanation_markers = ["Here", "This", "The solution", "We can", "Approach", "Explanation", "Note:"]
        found_markers = [m for m in explanation_markers if m in completion[:200]]
        print(f"  Explanation markers in first 200 chars: {found_markers}")
        print(f"  First 200 chars:\n    {repr(completion[:200])}")

        # ── Compare training vs eval prompt ──
        train_text = training_format(row)
        eval_prompt = eval_format(row, include_prelude=True)

        # The training prompt is everything before the assistant response
        train_prompt_part = train_text.split(CHAT_ASSISTANT)[0] + CHAT_ASSISTANT

        print("\n--- PROMPT COMPARISON ---")
        prompts_match = train_prompt_part == eval_prompt
        print(f"  Training prompt == Eval prompt: {prompts_match}")
        if not prompts_match:
            print(f"  TRAIN prompt:\n    {repr(train_prompt_part[:500])}")
            print(f"  EVAL  prompt:\n    {repr(eval_prompt[:500])}")
            # Find first difference
            for j, (a, b) in enumerate(zip(train_prompt_part, eval_prompt)):
                if a != b:
                    print(f"  First difference at char {j}: train={repr(a)} eval={repr(b)}")
                    print(f"  Context: train[{max(0,j-20)}:{j+20}] = {repr(train_prompt_part[max(0,j-20):j+20])}")
                    print(f"  Context: eval [{max(0,j-20)}:{j+20}] = {repr(eval_prompt[max(0,j-20):j+20])}")
                    break
            else:
                shorter = min(len(train_prompt_part), len(eval_prompt))
                print(f"  Strings match up to char {shorter}, but lengths differ: train={len(train_prompt_part)} eval={len(eval_prompt)}")

        print()

    # ── Summary stats across all samples ──
    print("=" * 80)
    print("SUMMARY ACROSS ALL SAMPLES")
    print("=" * 80)

    total = len(split)
    has_markdown = 0
    has_explanation = 0
    prompt_mismatches = 0
    has_prompt = 0
    has_query = 0
    has_both = 0

    for i in range(total):
        row = split[i]
        completion = row.get("completion", "")

        if "```" in completion:
            has_markdown += 1
        explanation_markers = ["Here", "This solution", "We can", "Approach:", "Explanation"]
        if any(m in completion[:200] for m in explanation_markers):
            has_explanation += 1

        p = (row.get("prompt") or "").strip()
        q = (row.get("query") or "").strip()
        if p:
            has_prompt += 1
        if q:
            has_query += 1
        if p and q:
            has_both += 1

        train_text = training_format(row)
        eval_prompt = eval_format(row, include_prelude=True)
        train_prompt_part = train_text.split(CHAT_ASSISTANT)[0] + CHAT_ASSISTANT
        if train_prompt_part != eval_prompt:
            prompt_mismatches += 1

    print(f"  Total samples: {total}")
    print(f"  Samples with 'prompt' field: {has_prompt} ({100*has_prompt/total:.1f}%)")
    print(f"  Samples with 'query' field: {has_query} ({100*has_query/total:.1f}%)")
    print(f"  Samples with both: {has_both} ({100*has_both/total:.1f}%)")
    print(f"  Prompt mismatches (train vs eval): {prompt_mismatches} ({100*prompt_mismatches/total:.1f}%)")
    print(f"  Completions with markdown (```): {has_markdown} ({100*has_markdown/total:.1f}%)")
    print(f"  Completions with explanation text: {has_explanation} ({100*has_explanation/total:.1f}%)")

    if has_markdown > 0:
        print(f"\n  ⚠️  {has_markdown} completions contain markdown code fences.")
        print(f"     The model will learn to output markdown-wrapped code.")
        print(f"     Make sure extract_code_from_completion handles this!")

    if has_explanation > 0:
        print(f"\n  ⚠️  {has_explanation} completions contain explanation text.")
        print(f"     The model will learn to explain before/after code.")
        print(f"     This makes code extraction harder and wastes tokens.")

    if prompt_mismatches > 0:
        print(f"\n  ⚠️  {prompt_mismatches} prompts differ between training and eval!")
        print(f"     This is a train/eval mismatch that will hurt performance.")


if __name__ == "__main__":
    main()