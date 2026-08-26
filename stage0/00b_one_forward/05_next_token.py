"""05 从 logits 到下一个 token。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch

from common import banner


def run(tokenizer, logits):
    banner("4. 从 logits 到下一个 token")
    last_logits = logits[0, -1, :]
    print(f"last_logits.shape = {tuple(last_logits.shape)}   # 词表里每个词一个分数")
    print(f"分数范围: {last_logits.min().item():.2f} ~ {last_logits.max().item():.2f}")

    next_id = last_logits.argmax(dim=-1).item()
    print()
    print(f"argmax 选出的 token id = {next_id}")
    print(f"对应文本 = {tokenizer.decode([next_id])!r}")

    probs = torch.softmax(last_logits.float(), dim=-1)
    top = torch.topk(probs, 5)
    print()
    print("排名前 5 的候选:")
    print(f"  {'排名':>4}  {'token id':>9}  {'概率':>8}  文本")
    for rank, (p, tid) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        print(f"  {rank:>4}  {tid:>9}  {p*100:>7.2f}%  {tokenizer.decode([tid])!r}")

    print()
    print("回答第四个问题：logits 经过 argmax（greedy）或 softmax + 抽样，变成一个 token id。")
    print("阶段 0 只用 argmax，因为它没有随机性，方便和 generate() 逐字对比。")
    return next_id


if __name__ == "__main__":
    from importlib import import_module

    tok, model = import_module("01_load").run()
    input_ids = import_module("02_tokenize").run(tok, model)
    logits = import_module("03_one_forward").run(model, input_ids)
    run(tok, logits)
