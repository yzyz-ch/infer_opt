"""03 一次 forward。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch

from common import Timer, banner


def run(model, input_ids):
    banner("2. 一次 forward")
    with torch.inference_mode():
        with Timer() as t_cold:
            model(input_ids=input_ids)
        with Timer() as t:
            outputs = model(input_ids=input_ids)

    print(f"第一次 forward（含预热）: {t_cold.elapsed*1000:7.2f} ms")
    print(f"第二次 forward（真实）  : {t.elapsed*1000:7.2f} ms")
    print(
        f"预热开销占了 {(1 - t.elapsed/t_cold.elapsed)*100:.0f}%，"
        "所以任何性能测量都要先预热再计时。"
    )
    print()
    print(f"返回对象类型: {type(outputs).__name__}")
    print(f"里面有哪些字段: {[k for k, v in outputs.items()]}")

    logits = outputs.logits
    print()
    print(f"logits.shape  = {tuple(logits.shape)}")
    print(
        f"              = [batch=1, seq_len={logits.shape[1]}, "
        f"vocab_size={logits.shape[2]}]"
    )
    print(f"logits.dtype  = {logits.dtype}")
    print(f"logits.device = {logits.device}")

    print()
    print("回答第二个问题：")
    print("  输入 = 一串 token id，形状 [batch, seq_len]")
    print("  输出 = 对词表里每个词的打分，形状 [batch, seq_len, vocab_size]")
    print()
    print(
        f"  这次一共产生了 {logits.numel():,} 个浮点数，"
        f"占 {logits.numel() * logits.element_size() / 1024**2:.1f} MB。"
    )
    print("  记住这个数，第 3 节会说明其中绝大部分是白算的。")
    return logits


if __name__ == "__main__":
    from importlib import import_module

    tok, model = import_module("01_load").run()
    input_ids = import_module("02_tokenize").run(tok, model)
    run(model, input_ids)
