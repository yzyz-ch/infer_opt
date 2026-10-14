"""03 每步耗时：短序列上看不到变慢。"""


def run(records):
    print("##### 3. 每步耗时：先别急着下结论 ###")
    n = len(records)
    times = [r["elapsed_ms"] for r in records]
    print(
        f"总共 {n} 步，累计 {sum(times)/1000:.2f} 秒，"
        f"平均每 token {sum(times)/n:.2f} ms"
    )

    if n >= 10:
        head = sum(times[:5]) / 5
        tail = sum(times[-5:]) / 5
        print()
        print(f"  前 5 步平均: {head:6.2f} ms  (序列长约 {records[2]['input_len']})")
        print(f"  后 5 步平均: {tail:6.2f} ms  (序列长约 {records[-3]['input_len']})")
        print(f"  变化: {(tail/head - 1)*100:+.1f}%")

    print()
    print("  每步耗时（每个 # 约 1ms）:")
    step_show = max(1, n // 20)
    for r in records[::step_show]:
        bar = "#" * int(r["elapsed_ms"])
        print(
            f"    step {r['step']:>3} len={r['input_len']:>3} "
            f"{r['elapsed_ms']:>6.2f}ms {bar}"
        )

    print("""
按常理推断，序列越长算得越多、应该越慢。但上面这组数字基本是平的，
甚至可能还略微下降，中间偶尔冒出几个尖峰（那多半是桌面程序在抢 GPU）。

这不是测错了，而是【测量尺度不对】。原因有两个：

  1. 序列才 20~50 个 token，Attention 那部分计算量小到可以忽略。
  2. 每跑一次 forward，要往 GPU 下发几百个 kernel（28 层，每层好几个）。
     这个固定开销加起来十几毫秒，把真正的计算时间盖住了。

也就是说，此刻的瓶颈不是"算得多"，而是"启动得多"和"权重读得多"。
这个模型只有 0.6B，每步都要把全部 1.1GB 权重从显存读一遍——
这件事的耗时和序列长度无关，却是当前的主要开销。

想看到增长趋势，得把序列拉长到固定开销盖不住的程度。见 04_length_sweep.py。
""")

if __name__ == "__main__":
    from importlib import import_module

    _tok, _model, _ids, records = import_module("01_run_loop").run()
    run(records)
