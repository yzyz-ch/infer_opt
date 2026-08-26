"""使用 KV Cache 的 greedy 解码，以及加速比测量。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch
from transformers import DynamicCache

from common import Timer, get_eos_ids

MAX_NEW_TOKENS = 64


def cached_greedy_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    verbose: bool = True,
):
    eos_ids = get_eos_ids(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    cache = DynamicCache()

    generated_ids = []
    records = []

    if verbose:
        print(
            f"{'步':>3} {'本轮喂入':>8} {'缓存长度':>8} {'阶段':>8} "
            f"{'耗时ms':>8}  文本"
        )
        print("-" * 74)

    with torch.inference_mode():
        with Timer() as t:
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        next_id = outputs.logits[0, -1, :].argmax(dim=-1).item()
        generated_ids.append(next_id)
        records.append(
            {
                "step": 0,
                "fed": input_ids.shape[1],
                "cache_len": cache.get_seq_length(),
                "phase": "prefill",
                "elapsed_ms": t.elapsed * 1000,
                "token_id": next_id,
            }
        )
        if verbose:
            print(
                f"{0:>3} {input_ids.shape[1]:>8} {cache.get_seq_length():>8} "
                f"{'prefill':>8} {t.elapsed*1000:>8.2f}  "
                f"{tokenizer.decode([next_id])!r}"
            )

        if next_id not in eos_ids:
            for step in range(1, max_new_tokens):
                cur = torch.tensor([[next_id]], device=input_ids.device)
                with Timer() as t:
                    outputs = model(
                        input_ids=cur,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                next_id = outputs.logits[0, -1, :].argmax(dim=-1).item()
                generated_ids.append(next_id)
                records.append(
                    {
                        "step": step,
                        "fed": 1,
                        "cache_len": cache.get_seq_length(),
                        "phase": "decode",
                        "elapsed_ms": t.elapsed * 1000,
                        "token_id": next_id,
                    }
                )
                if verbose:
                    print(
                        f"{step:>3} {1:>8} {cache.get_seq_length():>8} "
                        f"{'decode':>8} {t.elapsed*1000:>8.2f}  "
                        f"{tokenizer.decode([next_id])!r}"
                    )
                if next_id in eos_ids:
                    break

    if verbose:
        print("-" * 74)
    return generated_ids, records


def benchmark(model, context_len: int, n_new: int = 32):
    base = torch.randint(1000, 50000, (1, context_len), device=model.device)

    def run_no_cache():
        ids = base.clone()
        with torch.inference_mode():
            for _ in range(n_new):
                logits = model(input_ids=ids, logits_to_keep=1).logits
                nid = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, nid], dim=1)

    def run_cache():
        cache = DynamicCache()
        with torch.inference_mode():
            out = model(
                input_ids=base,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(n_new - 1):
                out = model(
                    input_ids=nid,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    run_no_cache()
    run_cache()
    with Timer() as t1:
        run_no_cache()
    with Timer() as t2:
        run_cache()
    return t1.elapsed, t2.elapsed


def kv_cache_bytes_per_token(model) -> int:
    cfg = model.config
    return (
        2
        * cfg.num_hidden_layers
        * cfg.num_key_value_heads
        * cfg.head_dim
        * torch.finfo(model.dtype).bits
        // 8
    )
