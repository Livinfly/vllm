import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
import os


model_name = "mini-qwen3-q8-kv2-1layer"

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, model_name)
os.makedirs(output_dir, exist_ok=True)

# fork_model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained(output_dir)

config = Qwen3Config(
    vocab_size=len(tokenizer),
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
)

model = Qwen3ForCausalLM(config)
model = model.to(dtype=torch.bfloat16)  # type: ignore

model.save_pretrained(output_dir)

print(f"model {model_name} save to {output_dir} successfully.")
