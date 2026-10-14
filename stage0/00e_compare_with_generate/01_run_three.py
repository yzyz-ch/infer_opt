"""01 三种方式各跑一遍。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch

from common import (
    DEFAULT_QUESTION,
    build_prompt,
    get_eos_ids,
    load_model_and_tokenizer,
)
from loops import load_loops

MAX_NEW_TOKENS = 64


def run(tokenizer=None, model=None):
    c_mod, d_mod = load_loops()
    if tokenizer is None or model is None:
        print("##### 0. 准备 ###")
        tokenizer, model = load_model_and_tokenizer()

    prompt = build_prompt(tokenizer, DEFAULT_QUESTION)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    eos_ids = get_eos_ids(model)

    print(f"\n问题: {DEFAULT_QUESTION}")
    print(f"prompt 长度: {prompt_ids.shape[1]} tokens")

    print("##### 1. 三种方式各跑一遍 ###")
    print("(1) 无缓存手写循环 —— 00c 的实现")
    ids_nocache, _ = c_mod.manual_greedy_decode(
        model, tokenizer, prompt, MAX_NEW_TOKENS, verbose=False
    )
    print("(2) 有缓存手写循环 —— 00d 的实现")
    ids_cache, _ = d_mod.cached_greedy_decode(
        model, tokenizer, prompt, MAX_NEW_TOKENS, verbose=False
    )
    print("(3) 官方 model.generate(do_sample=False)")
    with torch.inference_mode():
        out = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    ids_generate = out[0, prompt_ids.shape[1] :].tolist()

    print()
    print("注：如果上面出现关于 temperature / top_p / top_k 的警告，那是正常的。")
    print("    generation_config.json 里默认写了 do_sample=true 和这些采样参数，")
    print("    我们传了 do_sample=False，于是那些参数用不上，transformers 就提醒一句。")

    return {
        "tokenizer": tokenizer,
        "model": model,
        "prompt_ids": prompt_ids,
        "eos_ids": eos_ids,
        "ids_nocache": ids_nocache,
        "ids_cache": ids_cache,
        "ids_generate": ids_generate,
    }

if __name__ == "__main__":
    run()
