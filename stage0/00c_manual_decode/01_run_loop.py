"""01 手写生成循环（不使用 KV Cache）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch

from common import DEFAULT_QUESTION, banner, build_prompt, load_model_and_tokenizer
from greedy_loop import manual_greedy_decode


def warmup(model, tokenizer, prompt):
    with torch.inference_mode():
        warm = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        model(input_ids=warm)


def run(tokenizer=None, model=None):
    if tokenizer is None or model is None:
        banner("0. 准备")
        tokenizer, model = load_model_and_tokenizer()
        prompt = build_prompt(tokenizer, DEFAULT_QUESTION)
        print()
        print(f"问题: {DEFAULT_QUESTION}")
        warmup(model, tokenizer, prompt)
    else:
        prompt = build_prompt(tokenizer, DEFAULT_QUESTION)

    banner("1. 手写生成循环（不使用 KV Cache）")
    generated_ids, records = manual_greedy_decode(model, tokenizer, prompt)
    return tokenizer, model, generated_ids, records


if __name__ == "__main__":
    run()
