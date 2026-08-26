"""不使用 generate() 的 greedy 解码循环（故意不用 KV Cache）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch

from common import Timer, get_eos_ids

MAX_NEW_TOKENS = 64


def manual_greedy_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    verbose: bool = True,
):
    """返回 (generated_ids, records)。generated_ids 不含 prompt。"""
    eos_ids = get_eos_ids(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    generated_ids = []
    records = []

    if verbose:
        print(f"prompt 长度: {prompt_len} tokens")
        print(
            f"结束符 id  : {sorted(eos_ids)} "
            f"-> {[tokenizer.decode([i]) for i in sorted(eos_ids)]}"
        )
        print(f"最多生成   : {max_new_tokens} tokens")
        print()
        print(
            f"{'步':>3} {'输入长度':>6} {'logits shape':>22} "
            f"{'token id':>9} {'耗时ms':>8}  文本"
        )
        print("-" * 78)

    with torch.inference_mode():
        for step in range(max_new_tokens):
            with Timer() as t:
                outputs = model(input_ids=input_ids)

            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            next_id = next_token_logits.argmax(dim=-1).item()

            next_token = torch.tensor([[next_id]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            generated_ids.append(next_id)
            records.append(
                {
                    "step": step,
                    "input_len": logits.shape[1],
                    "logits_shape": tuple(logits.shape),
                    "token_id": next_id,
                    "elapsed_ms": t.elapsed * 1000,
                }
            )

            if verbose:
                piece = tokenizer.decode([next_id])
                print(
                    f"{step:>3} {logits.shape[1]:>6} "
                    f"{str(tuple(logits.shape)):>22} "
                    f"{next_id:>9} {t.elapsed*1000:>8.2f}  {piece!r}"
                )

            if next_id in eos_ids:
                if verbose:
                    print("-" * 78)
                    print(f"遇到结束符，第 {step} 步正常收尾")
                break
        else:
            if verbose:
                print("-" * 78)
                print(f"达到 {max_new_tokens} 上限，被强制截断（模型还想继续说）")

    return generated_ids, records
