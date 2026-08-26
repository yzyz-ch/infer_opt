"""
00a 已拆到目录 00a_tensor_basics/，按 01～08 逐个阅读。

运行其中一个:
    python 00a_tensor_basics/01_shape.py

一次跑完全部:
    python 00a_tensor_basics/run_all.py
"""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent / "00a_tensor_basics"
print("00a 已拆成若干小文件，见目录:", HERE)
print("下面按顺序跑 01～08。单独学习请打开对应文件。")
print()
runpy.run_path(str(HERE / "run_all.py"), run_name="__main__")
