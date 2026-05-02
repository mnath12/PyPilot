"""Quick script to show base model outputs (no LoRA)."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from code_execution import check_compilation, extract_code_from_completion, run_tests

CHAT_USER = "<|im_start|>user\n"
CHAT_ASSISTANT = "<|im_start|>assistant\n"
CHAT_END = "<|im_end|>"

def build_model_prompt(row, include_prelude=True):
    q = (row.get("query") or "").strip()
    prelude = (row.get("prompt") or "").strip()
    if include_prelude and prelude:
        user_content = prelude + "\n\n" + q
    else:
        user_content = q
    return f"{CHAT_USER}{user_content.strip()}{CHAT_END}\n{CHAT_ASSISTANT}"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Build stop token IDs
stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
for stop_token in ["<|im_end|>", "<|file_sep|>", "<|fim_prefix|>", "<|endoftext|>"]:
    token_id = tokenizer.convert_tokens_to_ids(stop_token)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        stop_token_ids.append(token_id)

print("Loading dataset...")
ds = load_from_disk("data/leetcode")

for i in range(3):
    row = ds["test"][i]
    print(f"\n{'='*80}")
    print(f"SAMPLE {i}: {row['task_id']} ({row['difficulty']})")
    print("="*80)

    prompt = build_model_prompt(row)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids if stop_token_ids else tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    print(f"\n--- Raw Output (first 600 chars) ---")
    print(generated[:600])

    extracted = extract_code_from_completion(generated, row.get("starter_code", ""))
    print(f"\n--- Extracted Code ---")
    print(extracted[:500])

    compiles, err = check_compilation(extracted)
    print(f"\n--- Compiles: {compiles} ---")
    if not compiles:
        print(f"Error: {err}")

    if compiles:
        passed, test_err = run_tests(extracted, row.get("test", ""), row.get("entry_point", ""), timeout=10)
        print(f"--- Tests Pass: {passed} ---")
        if not passed:
            print(f"Error: {test_err[:300] if test_err else 'Unknown'}")

print("\nDone!")
