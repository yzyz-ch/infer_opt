"""02 三者对比。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

from common import banner


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run(tokenizer, ids_nocache, ids_cache, ids_generate):
    banner("2. 三者对比")
    rows = [
        ("无缓存手写", ids_nocache),
        ("有缓存手写", ids_cache),
        ("generate()", ids_generate),
    ]
    print(f"  {'实现':<12} {'token数':>7}  文本")
    print("  " + "-" * 74)
    for name, ids in rows:
        print(
            f"  {name:<12} {len(ids):>7}  "
            f"{tokenizer.decode(ids, skip_special_tokens=True)}"
        )

    print()
    d_nc = first_diff(ids_nocache, ids_generate)
    d_c = first_diff(ids_cache, ids_generate)
    print(
        "  无缓存手写 vs generate(): "
        f"{'完全一致' if d_nc is None else f'第 {d_nc} 步开始分歧'}"
    )
    print(
        "  有缓存手写 vs generate(): "
        f"{'完全一致' if d_c is None else f'第 {d_c} 步开始分歧'}"
    )

    print("""
这个结果第一眼很反直觉：明明"无缓存"那版逻辑最朴素、最不容易写错，
反而对不上；"有缓存"那版复杂得多，却完全一致。

原因不在逻辑，在数值路径。
generate() 内部默认是开 KV Cache 的，所以它和"有缓存手写"走的是同一条数值路径。
逻辑正确 + 数值路径相同 = 逐 token 一致。

而"无缓存"那版逻辑同样正确，只是数值路径不同，于是结果允许有细微差别。
""")
    return d_nc, d_c


if __name__ == "__main__":
    from importlib import import_module

    ctx = import_module("01_run_three").run()
    run(
        ctx["tokenizer"],
        ctx["ids_nocache"],
        ctx["ids_cache"],
        ctx["ids_generate"],
    )
