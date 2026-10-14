"""03 把分歧的那一步拆开看。"""

import torch
from transformers import DynamicCache


def show_tokens(tokenizer, ids, limit=12):
    pieces = [tokenizer.decode([i]) for i in ids[:limit]]
    tail = " ..." if len(ids) > limit else ""
    return " | ".join(p.replace("\n", "\\n") for p in pieces) + tail


def run(tokenizer, model, prompt_ids, ids_nocache, d_nc):
    print("##### 3. 把分歧的那一步拆开看 ###")
    if d_nc is None:
        print("这次没有出现分歧（有一定偶然性），跳过本节。")
        print("多换几个问题试试，很容易复现。")
        return

    k = d_nc
    print(f"第 {k} 步之前两边完全相同，就从这一步开始查。")
    print(
        f"  共同前缀（最后几个词）: "
        f"...{show_tokens(tokenizer, ids_nocache[max(0, k-5):k])}"
    )
    print()

    prefix = (
        torch.cat(
            [
                prompt_ids,
                torch.tensor(
                    [ids_nocache[:k]], device=model.device, dtype=prompt_ids.dtype
                ),
            ],
            dim=1,
        )
        if k > 0
        else prompt_ids
    )

    with torch.inference_mode():
        logits_a = model(input_ids=prefix).logits[0, -1, :].float()
        cache = DynamicCache()
        model(input_ids=prefix[:, :-1], past_key_values=cache, use_cache=True)
        logits_b = (
            model(input_ids=prefix[:, -1:], past_key_values=cache, use_cache=True)
            .logits[0, -1, :]
            .float()
        )

    top_a = torch.topk(logits_a, 3)
    top_b = torch.topk(logits_b, 3)

    print("  两条数值路径给出的前 3 名:")
    print(f"    {'排名':>4}  {'整条前向(00c)':<28} {'缓存解码(00d)':<28}")
    for r in range(3):
        ta = (
            f"{tokenizer.decode([top_a.indices[r].item()])!r} "
            f"{top_a.values[r].item():.4f}"
        )
        tb = (
            f"{tokenizer.decode([top_b.indices[r].item()])!r} "
            f"{top_b.values[r].item():.4f}"
        )
        print(f"    {r+1:>4}  {ta:<28} {tb:<28}")

    margin_a = (top_a.values[0] - top_a.values[1]).item()
    diff = (logits_a - logits_b).abs().max().item()
    print()
    print(f"  两条路径 logits 的最大差异: {diff:.4f}")
    print(f"  第一名和第二名的分数差距  : {margin_a:.4f}")
    print()
    if margin_a < diff:
        print("  差距 < 数值误差，所以谁排第一完全由浮点舍入决定，翻转是必然的。")
    else:
        print("  这一步差距虽然大于误差，但前面某处的微小差异已经改变了上下文。")

    print("""
  结论：bf16 只有 8 位尾数，不同矩阵形状会走不同 kernel，累加顺序不同，
  logits 就会有 0.0x 量级的抖动。平时无所谓，但只要某一步的前两名咬得很紧，
  这点抖动就足以让 argmax 翻转。而生成是自回归的，一步走岔，后面全岔。

  这不是 bug，是 bf16 推理的固有特性。
  真实项目里做正确性验证时，判据通常是这样定的：
      - 逐 token 完全一致（要求两边数值路径相同，最严格）
      - top-1 一致率 / 困惑度差异在阈值内（路径不同时用这个）
      - 人工评测输出质量（最终兜底）
  拿"文字一模一样"当唯一标准，会冤枉很多其实正确的实现。
""")

if __name__ == "__main__":
    from importlib import import_module

    ctx = import_module("01_run_three").run()
    d_nc, _d_c = import_module("02_compare").run(
        ctx["tokenizer"],
        ctx["ids_nocache"],
        ctx["ids_cache"],
        ctx["ids_generate"],
    )
    run(
        ctx["tokenizer"],
        ctx["model"],
        ctx["prompt_ids"],
        ctx["ids_nocache"],
        d_nc,
    )
