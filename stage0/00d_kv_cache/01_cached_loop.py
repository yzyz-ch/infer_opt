"""01 带 KV Cache 的生成循环。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch
from transformers import DynamicCache

from cached_loop import cached_greedy_decode
from common import DEFAULT_QUESTION, build_prompt, load_model_and_tokenizer


def warmup(model, tokenizer, prompt):
    with torch.inference_mode():
        warm = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        warm_cache = DynamicCache()
        model(input_ids=warm, past_key_values=warm_cache, use_cache=True)
        one = torch.tensor([[warm[0, -1].item()]], device=model.device)
        model(input_ids=one, past_key_values=warm_cache, use_cache=True)


def run(tokenizer=None, model=None):
    if tokenizer is None or model is None:
        print("##### 0. 准备 ###")
        tokenizer, model = load_model_and_tokenizer()
        prompt = build_prompt(tokenizer, DEFAULT_QUESTION)
        warmup(model, tokenizer, prompt)
    else:
        prompt = build_prompt(tokenizer, DEFAULT_QUESTION)

    print("##### 1. 带 KV Cache 的生成循环 ###")
    print(f"问题: {DEFAULT_QUESTION}")
    print()
    generated_ids, records = cached_greedy_decode(model, tokenizer, prompt)
    print()
    print("生成结果:")
    print(f"  {tokenizer.decode(generated_ids, skip_special_tokens=True)}")
    return tokenizer, model, generated_ids, records

if __name__ == "__main__":
    run()
