import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-Coder-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "def gcd(a, b):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.8,
    do_sample=True
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
