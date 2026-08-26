"""
00e 已拆到目录 00e_compare_with_generate/。

单独阅读:
    python 00e_compare_with_generate/01_run_three.py

一次跑完全部:
    python 00e_compare_with_generate/run_all.py
"""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent / "00e_compare_with_generate"
print("00e 已拆成若干小文件，见目录:", HERE)
print()
runpy.run_path(str(HERE / "run_all.py"), run_name="__main__")
