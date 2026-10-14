"""04 换个尺度：forward 耗时 vs 序列长度。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch

from common import Timer, load_model_and_tokenizer


def run(model):
    print("##### 4. 换个尺度：forward 耗时 vs 序列长度 ###")
    print("用随机 token 构造不同长度的输入，各测 10 次取平均：")
    print()
    print(f"  {'序列长度':>8} {'forward耗时':>12} {'每token':>10}   相对 16 长度")
    print("  " + "-" * 56)

    sweep = [16, 64, 256, 512, 1024, 2048, 4096]
    base_ms = None
    with torch.inference_mode():
        for length in sweep:
            fake = torch.randint(1000, 50000, (1, length), device=model.device)
            for _ in range(3):
                model(input_ids=fake, logits_to_keep=1)
            with Timer() as t:
                for _ in range(10):
                    model(input_ids=fake, logits_to_keep=1)
            ms = t.elapsed * 1000 / 10
            base_ms = base_ms or ms
            print(
                f"  {length:>8} {ms:>10.2f} ms {ms/length:>9.4f} "
                f"     {ms/base_ms:>5.2f}x"
            )

    print("""
现在趋势就清楚了：

  - 512 以内，耗时几乎是一条水平线。固定开销主导，跟序列长度没关系。
  - 1024 往上开始明显爬升。
  - 4096 相比 2048，长度翻倍但耗时涨了两倍多——这是超线性。
    因为 Attention 要让每个 token 和前面所有 token 算相关性，
    计算量按 O(n^2) 增长，长上下文时它会变成主角。

顺带一个很实用的教训：性能结论必须绑定测量条件。
"KV Cache 能加速"这句话在 4096 上下文是对的，在 50 个 token 上却是错的。
00d 会用数字把这句话验证一遍。
""")

if __name__ == "__main__":
    _tok, model = load_model_and_tokenizer()
    run(model)
