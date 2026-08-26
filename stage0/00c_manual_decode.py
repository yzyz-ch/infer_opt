"""
00c 已拆到目录 00c_manual_decode/。

单独阅读:
    python 00c_manual_decode/01_run_loop.py

一次跑完全部:
    python 00c_manual_decode/run_all.py
"""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent / "00c_manual_decode"
print("00c 已拆成若干小文件，见目录:", HERE)
print()
runpy.run_path(str(HERE / "run_all.py"), run_name="__main__")
