"""
00b 已拆到目录 00b_one_forward/。

单独阅读:
    python 00b_one_forward/02_tokenize.py

一次跑完全部（只加载一次模型）:
    python 00b_one_forward/run_all.py
"""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent / "00b_one_forward"
print("00b 已拆成若干小文件，见目录:", HERE)
print()
runpy.run_path(str(HERE / "run_all.py"), run_name="__main__")
