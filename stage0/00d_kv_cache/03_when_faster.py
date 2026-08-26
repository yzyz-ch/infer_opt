"""03 什么时候 KV Cache 才真的加速。这一节最慢。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

from cached_loop import benchmark
from common import banner, load_model_and_tokenizer


def run(model):
    banner("3. 关键问题：什么时候才真的加速")
    print("在不同上下文长度下，各生成 32 个 token，对比两种方式：")
    print()
    print(f"  {'上下文长度':>10} {'重算全部':>12} {'用缓存':>12} {'加速比':>8}")
    print("  " + "-" * 48)

    for ctx in (128, 512, 1024, 2048, 4096):
        t_no, t_yes = benchmark(model, ctx, n_new=32)
        print(
            f"  {ctx:>10} {t_no:>10.2f} s {t_yes:>10.2f} s "
            f"{t_no/t_yes:>7.2f}x"
        )

    print("""
读这张表要注意：

  - 上下文很短时，加速比接近 1，甚至可能小于 1（缓存反而略慢）。
    因为此时瓶颈是 kernel 启动和读权重，而 KV Cache 一点也帮不上这两件事。
    它还额外带来了每层维护缓存的 Python 开销，所以可能倒亏一点。

  - 上下文越长，加速比越大。因为"重算全部"的代价随长度线性甚至超线性增长，
    而"用缓存"的每步代价基本不变。

真实服务场景的上下文动辄几千上万 token，所以 KV Cache 是必备的。
但你要记住它成立的前提，而不是把"KV Cache 更快"当成一条无条件的教条。

这也是这个阶段最该带走的思维方式：
任何优化都有生效范围，先量出范围，再决定用不用。
""")


if __name__ == "__main__":
    _tok, model = load_model_and_tokenizer()
    run(model)
