import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = str(
    Path(__file__).resolve().parents[3] / "local" / "models" / "Qwen3-0.6B"
)
PROMPT = "请用三句话解释什么是大语言模型推理。"


def cuda_mem():
    if not torch.cuda.is_available():
        return "cuda not available"
    # PyTorch 以字节为单位统计 CUDA 峰值显存；这里转换成 GiB，便于阅读日志。
    # allocated 表示张量实际使用的显存，reserved 包含 PyTorch 缓存池预留的显存。
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    return f"max_allocated={allocated:.2f}GB, max_reserved={reserved:.2f}GB"


def main():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() 

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    t1 = time.perf_counter()

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t2 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("=" * 80)
    print("[HF] output:")
    print(result)
    print("=" * 80)
    print(f"[HF] load_time_sec: {t1 - t0:.3f}")
    print(f"[HF] generate_time_sec: {t3 - t2:.3f}")
    print(f"[HF] new_tokens: {len(new_tokens)}")
    print(f"[HF] avg_sec_per_token: {(t3 - t2) / max(len(new_tokens), 1):.4f}")
    print(f"[HF] cuda_mem: {cuda_mem()}")


if __name__ == "__main__":
    main()