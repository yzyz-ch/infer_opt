import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

MODEL_PATH = str(
    Path(__file__).resolve().parents[3] / "local" / "models" / "Qwen3-0.6B"
)
PROMPT = "请用三句话解释什么是大语言模型推理。"


def cuda_mem():
    if not torch.cuda.is_available():
        return "cuda not available"
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    return f"max_allocated={allocated:.2f}GB, max_reserved={reserved:.2f}GB"


def main():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True, 
    )

    t0 = time.perf_counter()
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    t1 = time.perf_counter()

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )

    t2 = time.perf_counter()
    outputs = llm.generate([text], sampling_params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    result = outputs[0]["text"]
    # 这里只是粗略估计 token 数；后面学习 Runtime 时再深入。
    token_count = len(tokenizer(result).input_ids)

    print("=" * 80)
    print("[nano-vLLM] output:")
    print(result)
    print("=" * 80)
    print(f"[nano-vLLM] load_time_sec: {t1 - t0:.3f}")
    print(f"[nano-vLLM] generate_time_sec: {t3 - t2:.3f}")
    print(f"[nano-vLLM] approx_new_tokens: {token_count}")
    print(f"[nano-vLLM] avg_sec_per_token: {(t3 - t2) / max(token_count, 1):.4f}")
    print(f"[nano-vLLM] cuda_mem: {cuda_mem()}")


if __name__ == "__main__":
    main()