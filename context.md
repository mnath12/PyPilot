# PyPilot / LLM Coding Assistant — Cursor Context

## What this project is
PyPilot is an LLM-based coding assistant whose primary deliverable is: generate small **Python programs** and solve **LeetCode-style problems** from natural-language prompts. It should be usable in both (1) a **VSCode extension** and (2) a **web-based Python editor**. :contentReference[oaicite:0]{index=0}

The “core engine” is a **prebuilt code model** (initially Qwen-2.5-Coder per the write-up), which we “supercharge” using structured methods rather than attempting to directly hand-edit weights. 


## Current plan (explicit roadmap)
We will proceed in this order:

1. **Baseline evaluation of the base model** (no finetuning, no RL)  
   - Establish reference metrics (pass rate, compile rate, latency, etc.).
2. **Finetuning (SFT / PEFT-first)**  
   - Improve performance on our target task distribution with supervised training.
3. **RL stage (PPO-style)**  
   - Treat code generation as an MDP and optimize an explicit reward objective.

This ordering is intentional: we want clean attribution of gains and stable iteration.

---

## Theoretical framing (MDP for code generation)
We model code generation as a discrete-time Markov Decision Process (MDP) where **each token emission is one time step**. :contentReference[oaicite:2]{index=2}

- **State** at time *t*: the full prefix of generated tokens  
  \[
  X_t = (x_0, x_1, \dots, x_t)
  \]
- **Action**: choose the next token \(x_{t+1}\) from vocabulary \(V\).
- **Transition**: deterministic “append-token” transition.
- **Terminal condition**: EOS token or max length reached.
- **Reward (sparse terminal reward)**: collected only when generation ends:  
  +1 if the program compiles & passes tests, −1 otherwise (as currently defined). :contentReference[oaicite:3]{index=3}

The policy is the model itself:
\[
\pi_\theta(a \mid X_t) = \text{softmax}(f_\theta(X_t))_a
\]
and the RL objective is to maximize expected discounted return (often start with \(\gamma = 1\)). :contentReference[oaicite:4]{index=4}

---

## “Supercharge” methods we may incorporate
These are candidate research directions for improving the core engine beyond plain SFT:

1. **Stochastic-process / RL view (primary)**  
   - Codegen as MDP + PPO-style optimization of expected return. 
2. **Graph-theoretic view (AST features)**  
   - Build AST graphs of solutions; embed with graph transformers / diffusion (heat-kernel style features). :contentReference[oaicite:6]{index=6}
3. **Risk-adjusted beam search (decoding-time steering)**  
   - Re-rank beams by likelihood plus an online “will this compile/pass tests” estimate to steer toward safe completions. :contentReference[oaicite:7]{index=7}
4. **Dataset selection as submodular optimization**  
   - Greedy selection of a compact but diverse training set (coverage of tokens/control-flow motifs) to cut finetune cost. :contentReference[oaicite:8]{index=8}

---

## What “good” outputs look like (project targets)
When Cursor asks the model for a solution, prefer outputs that are:

- **Self-contained** Python (single file unless explicitly asked otherwise)
- Deterministic + testable (no hidden state, no “assume X exists”)
- Handles edge cases
- Clear time/space complexity
- Minimal dependencies (stdlib unless asked)
- Includes quick sanity tests when helpful (but not huge test suites)

---

## Evaluation harness (baseline → SFT → RL)
### Baseline (base model)
Goal: measure “where we are” before training.
Recommended metrics:
- **pass@1** on a held-out prompt/test set
- **compile / runtime error rate**
- **average tokens**, latency, and timeouts
- **format correctness** (function signature, I/O contract)

### Finetuning (SFT / PEFT-first)
Goal: shift the policy toward correct solution patterns.
- Start with SFT on (prompt → reference solution) pairs
- Prefer PEFT (LoRA/adapters) initially for faster iteration

### RL (PPO-style)
Goal: optimize reward tied to actual correctness.
- Reward computed by executing unit tests / hidden tests
- Start with sparse terminal reward (per write-up), then consider shaping later
- Carefully log: reward distribution, KL to reference policy, and regression in style/length

---

## Notes for Cursor usage (how to behave when editing/adding code)
When generating or modifying repository code:
- Keep changes small and localized; avoid broad refactors unless necessary.
- Preserve existing APIs and entry points.
- Add docstrings + type hints in new code.
- Prefer explicit, readable Python over clever tricks.
- If introducing training code, include:
  - fixed random seeds
  - clear config surface (YAML/JSON or dataclass)
  - logging hooks (loss, eval metrics, checkpoints)

---

## Suggested repo skeleton (high-level)
- `models/` — model loading, tokenizer, PEFT adapters
- `data/` — dataset loaders, prompt formatting, train/val splits
- `eval/` — offline evaluation harness + metrics + reports
- `train_sft/` — supervised training scripts
- `train_rl/` — PPO loop, reward runner (compile/tests sandbox)
- `decoding/` — beam search, risk-adjusted decoding experiments
- `tools/` — sandbox execution, test running, safety/timeouts
- `vscode_ext/` + `web/` — frontends

---

## “North Star” objective
Build a coding assistant that is measurably better than the base model on our target distribution, with improvements attributable to:
1) clean baseline measurement, then
2) supervised finetuning gains, then
3) RL optimization tied directly to compilation and test success. 
