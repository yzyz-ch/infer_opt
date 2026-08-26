"""按顺序跑完 01～08。在 stage0 目录下执行: python 00a_tensor_basics/run_all.py"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

scripts = sorted(HERE.glob("[0-9][0-9]_*.py"))

for path in scripts:
    print("\n" + "#" * 78)
    print(f"# {path.name}")
    print("#" * 78)
    runpy.run_path(str(path), run_name="__main__")
