"""
00d 已拆到目录 00d_kv_cache/。

单独阅读:
    python 00d_kv_cache/01_cached_loop.py

一次跑完全部:
    python 00d_kv_cache/run_all.py
"""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent / "00d_kv_cache"
print("00d 已拆成若干小文件，见目录:", HERE)
print()
runpy.run_path(str(HERE / "run_all.py"), run_name="__main__")
