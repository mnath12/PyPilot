"""
PyPilot – PPO fine-tuning script
================================
• Computes baseline compile / pass rate on a sample of tasks
• Fine-tunes the model with PPO, using the sandbox-based reward
• Re-evaluates the same sample to show improvement
----------------------------------------------------------------
Expectations
------------
❶ You already have:
   • sandbox.py   (provides run_tests)
   • the HuggingFace dataset "newfacade/LeetCodeDataset"
   • transformers, datasets, trl et al. installed
❷ This works on CPU (0.5 B model) – flip USE_CPU=False for GPU.
❸ Adjust TOTAL_EPISODES, BATCH_SIZE, etc. for longer runs.
"""

import os, random, textwrap, traceback, ast, types
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from datasets import load_dataset
from tqdm.auto import tqdm

# ────────────────────────────────────────────────────────────────
# 0. Global switches – edit as needed
# ────────────────────────────────────────────────────────────────
USE_CPU        = True                      # False → GPU 7 B model
DEVICE         = torch.device("cpu" if USE_CPU else "cuda")
BASE_MODEL     = "Qwen/Qwen2.5-Coder-0.5B" if USE_CPU else "Qwen/Qwen2.5-Coder-7B-Instruct"
TOTAL_EPISODES = 20 if USE_CPU else 200    # PPO iterations
BATCH_SIZE     = 1  if USE_CPU else 4
SAMPLE_SIZE    = 20 if USE_CPU else 100    # evaluation subset
GEN_TOKENS     = 256                       # max new tokens per generation
TIMEOUT        = 2                         # sec per sandbox run

# ────────────────────────────────────────────────────────────────
# 1. Load model / tokenizer
# ────────────────────────────────────────────────────────────────
print(f"▶ Loading {BASE_MODEL} on {DEVICE} …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model.to(DEVICE).eval()

# ────────────────────────────────────────────────────────────────
# 2. Dataset
# ────────────────────────────────────────────────────────────────
leet_ds = load_dataset("newfacade/LeetCodeDataset", split="train")
eval_subset = random.sample(list(leet_ds), min(SAMPLE_SIZE, len(leet_ds)))

# ────────────────────────────────────────────────────────────────
# 3. Prompt builders
# ────────────────────────────────────────────────────────────────
SYS_PROMPT = (
    "You are PyPilot, an expert competitive programming assistant. "
    "Extend the code snippet by finishing class Solution based on all provided "
    "instructions and information.\n"
)

def build_prompt(example: Dict) -> str:
    """Return the text fed to the LM (no fenced code)."""
    return textwrap.dedent(f"""{SYS_PROMPT}
    # Problem: {example['problem_description']}

    # Starter code:
    {example['starter_code']}

    # Fill in class Solution:
    """).strip()

# ────────────────────────────────────────────────────────────────
# 4. Sandbox-based compile + test check
# ────────────────────────────────────────────────────────────────
from sandbox import run_tests     # make sure sandbox.py is importable!

def get_candidate_expr(starter: str) -> str:
    """
    Heuristic: return "Solution().<first_method>" so tests can call it.
    Fallback to 'candidate' if no class detected.
    """
    for line in starter.splitlines():
        line = line.strip()
        if line.startswith("def "):
            fn = line.split("(")[0][4:]
            return fn                       # simple script w/ top-level function
        if line.startswith("class Solution"):
            # grab first 'def' **after** class Solution
            continue
    # default (tests will need 'candidate' defined in user code)
    return "candidate"

def evaluate_one(example: Dict, code: str) -> Dict[str, bool]:
    candidate_expr = "Solution()." + example["entry_point"] if "entry_point" in example else "candidate"
    return run_tests(
        user_src   = code,
        tests_src  = example["unit_tests"],
        candidate_expr = candidate_expr,
        timeout    = TIMEOUT,
    )

# ────────────────────────────────────────────────────────────────
# 5. Helper: LM generation (strip prompt prefix)
# ────────────────────────────────────────────────────────────────
def gen_completion(model, prompt: str) -> str:
    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out_ids = model.generate(
            **tokens,
            max_new_tokens=GEN_TOKENS,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    # keep only completion
    return out_text[len(prompt):].strip()

# ────────────────────────────────────────────────────────────────
# 6. Baseline evaluation
# ────────────────────────────────────────────────────────────────
def calc_metrics(model, subset: List[Dict]) -> Dict[str, float]:
    passes = compiles = 0
    for ex in tqdm(subset, desc="Evaluating"):
        prompt = build_prompt(ex)
        code   = gen_completion(model, prompt)
        res    = evaluate_one(ex, code)
        passes   += res["passed"]
        compiles += res["compile"]
    n = len(subset)
    return {"compile_rate": compiles / n, "pass_rate": passes / n}

print("\n🔎 Baseline evaluation …")
baseline = calc_metrics(base_model, eval_subset)
print("📊 Baseline:", baseline)

# ────────────────────────────────────────────────────────────────
# 7. PPO setup
# ────────────────────────────────────────────────────────────────
ppo_cfg = PPOConfig(
    model_name      = BASE_MODEL,
    learning_rate   = 2e-5 if USE_CPU else 5e-6,
    batch_size      = BATCH_SIZE,
    mini_batch_size = BATCH_SIZE,
    ppo_epochs      = 1 if USE_CPU else 4,
    total_episodes  = TOTAL_EPISODES, 
)
# value head wrapper
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model).to(DEVICE)
ppo_trainer  = PPOTrainer(config=ppo_cfg, model=policy_model, tokenizer=tokenizer)

# ────────────────────────────────────────────────────────────────
# 8. PPO training loop
# ────────────────────────────────────────────────────────────────
print("\n🚀 Starting PPO …")
for ep in range(TOTAL_EPISODES):
    batch = random.sample(list(leet_ds), BATCH_SIZE)
    prompts = [build_prompt(ex) for ex in batch]

    # model responses
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    gen_ids   = policy_model.generate(**tokenized, max_new_tokens=GEN_TOKENS, temperature=0.2)
    responses = []
    for ids, p in zip(gen_ids, prompts):
        txt = tokenizer.decode(ids, skip_special_tokens=True)
        responses.append(txt[len(p):].strip())

    # rewards
    rewards = []
    for code, ex in zip(responses, batch):
        out = evaluate_one(ex, code)
        rewards.append(1.0 if out["passed"] else -1.0)

    # PPO step
    ppo_trainer.step(prompts, responses, rewards)

    if (ep + 1) % 5 == 0 or ep == TOTAL_EPISODES - 1:
        mean_r = sum(rewards) / len(rewards)
        print(f"  • Episode {ep+1}/{TOTAL_EPISODES}  |  mean reward = {mean_r:+.2f}")

# save
ppo_trainer.save_pretrained("pypilot_ppo_finetune")

# ────────────────────────────────────────────────────────────────
# 9. Post-training evaluation
# ────────────────────────────────────────────────────────────────
print("\n📥 Loading fine-tuned model …")
ft_model = AutoModelForCausalLM.from_pretrained("pypilot_ppo_finetune").to(DEVICE).eval()

print("\n🔎 Post-training evaluation …")
after = calc_metrics(ft_model, eval_subset)
print("📊 After RL:", after)

print("\n✅ Done.  Baseline vs. after-training:")
print(f"    compile_rate: {baseline['compile_rate']:.2%} → {after['compile_rate']:.2%}")
print(f"    pass_rate   : {baseline['pass_rate']:.2%} → {after['pass_rate']:.2%}")
