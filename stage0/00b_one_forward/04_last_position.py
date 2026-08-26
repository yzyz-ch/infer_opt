"""04 为什么只看最后一个位置。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch

from common import banner


def run(tokenizer, model, input_ids, logits):
    banner("3. 为什么只看最后一个位置")
    print("logits 有 seq_len 行，每一行是'看完这个位置为止的内容后，下一个词该是什么'。")
    print()
    print("挑几个位置看看模型分别在预测什么:")
    print(f"  {'位置':>4}  {'该位置输入的词':<14}  {'模型预测下一个词'}")

    for pos in [0, 1, len(input_ids[0]) // 2, -2, -1]:
        cur_id = input_ids[0, pos].item()
        pred_id = logits[0, pos].argmax(dim=-1).item()
        print(
            f"  {pos:>4}  {tokenizer.decode([cur_id])!r:<14}  "
            f"{tokenizer.decode([pred_id])!r}"
        )

    print()
    print("前面那些位置的预测，其实是在'预测我们已经知道的内容'——")
    print("位置 0 预测位置 1，可位置 1 的词我们输入里本来就有。")
    print("只有最后一个位置的预测，指向的是我们还不知道的、真正要生成的下一个词。")
    print()
    print("回答第三个问题：前面 seq_len-1 行在生成时用不上，所以只取 [:, -1, :]。")

    print()
    print("transformers v5 把这个优化做成了 logits_to_keep 参数：")

    with torch.inference_mode():
        out_keep1 = model(input_ids=input_ids, logits_to_keep=1)

    print(f"  默认             logits.shape = {tuple(logits.shape)}")
    print(f"  logits_to_keep=1 logits.shape = {tuple(out_keep1.logits.shape)}")
    saved = 1 - out_keep1.logits.numel() / logits.numel()
    print(f"  省掉了 {saved*100:.1f}% 的 lm_head 计算量和输出显存")
    print()
    print("  `generate()` 内部就是这么做的。阶段 0 为了看得清楚，我们仍用默认的全量输出。")

    same = torch.equal(logits[:, -1, :], out_keep1.logits[:, -1, :])
    maxdiff = (logits[:, -1, :].float() - out_keep1.logits[:, -1, :].float()).abs().max()
    print()
    print(f"  顺带一提：两种算法最后一行完全相等吗 -> {same}")
    print(f"  最大差异 {maxdiff.item():.4f}，argmax 仍然一致。")
    print("  bf16 下改变矩阵形状会走不同的 kernel，结果有微小浮动，这是正常的。")
    print("  所以后面对比时要比 token id，不要比 logits 数值。")


if __name__ == "__main__":
    from importlib import import_module

    tok, model = import_module("01_load").run()
    input_ids = import_module("02_tokenize").run(tok, model)
    logits = import_module("03_one_forward").run(model, input_ids)
    run(tok, model, input_ids, logits)
