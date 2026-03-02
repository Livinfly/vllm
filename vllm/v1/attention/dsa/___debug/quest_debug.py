import os
import torch
from vllm import LLM, SamplingParams

model_name = "mini-qwen3-q8-kv2-2layers"
model_name = "mini-qwen3-q8-kv2-1layer"

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, model_name)

BLOCK_SIZE = 16 

ENFORCE_EAGER = True 

GPU_UTILIZATION = 0.15

def run_debug_session():
    print(f"--- Starting vLLM Debug Session ---")
    print(f"Target Model: {model_name}")
    print(f"Block Size: {BLOCK_SIZE}")
    print(f"Eager Mode: {ENFORCE_EAGER} (Crucial for pdb/print)")

    llm = LLM(
        model=model_path,
        enforce_eager=ENFORCE_EAGER, 
        gpu_memory_utilization=GPU_UTILIZATION,
        tensor_parallel_size=1,
        attention_config={"backend":"FLASH_ATTN"},
    )

    long_prompt = "Artificial intelligence is a field " * 20
    # batch_size = 2
    short_prompt = "Considering the rapid advancement of artificial intelligence and its impact on the global job market, which specific skills, such as critical thinking, creative problem-solving, or emotional intelligence, do you believe are most essential to cultivate now to ensure long-term career adaptability, job security, and personal fulfillment in the next decade"
    
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=3,
        min_tokens=3,
    )

    print("\n--- Triggering Inference ---")
    print("Check your console for prints from QuestSelectStrategy...")

    outputs = llm.generate([long_prompt, short_prompt], sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"\nGenerated: {generated_text!r}")

if __name__ == "__main__":
    # torch.set_printoptions(profile="full", linewidth=200, threshold=10000)
    run_debug_session()