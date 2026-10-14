"""02 文本 -> token id。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

from common import DEFAULT_QUESTION, build_prompt


def run(tokenizer, model):
    print("##### 1. 文本 -> token id ###")
    print(f"你的问题原文:\n  {DEFAULT_QUESTION}")

    prompt = build_prompt(tokenizer, DEFAULT_QUESTION)
    print()
    print("套上对话模板之后，模型真正看到的文本:")
    print(f"  {prompt!r}")
    print()
    print("  注意 <|im_start|> / <|im_end|> 这些是特殊 token，不是普通文字。")
    print("  它们告诉模型'这段是用户说的''该你回答了'。")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    print()
    print(
        f"input_ids.shape  = {tuple(input_ids.shape)}   "
        f"# [batch=1, seq_len={input_ids.shape[1]}]"
    )
    print(f"input_ids.dtype  = {input_ids.dtype}          # id 是整数，不是浮点")
    print(f"input_ids.device = {input_ids.device}")

    print()
    print("逐个看看这句话被切成了什么（前 25 个）:")
    print(f"  {'序号':>4}  {'token id':>9}  文本")
    for i, tid in enumerate(input_ids[0].tolist()[:25]):
        piece = tokenizer.decode([tid])
        print(f"  {i:>4}  {tid:>9}  {piece!r}")

    print()
    print("回答第一个问题：文本在 tokenizer(...) 这一步变成 token id，")
    print("在这之后，模型内部再也见不到'文字'，只有整数和浮点数。")
    return input_ids

if __name__ == "__main__":
    from importlib import import_module

    tok, model = import_module("01_load").run()
    run(tok, model)
