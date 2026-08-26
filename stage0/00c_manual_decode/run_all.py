"""按顺序跑完 00c。在 stage0 目录下: python 00c_manual_decode/run_all.py"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _path  # noqa: F401

from importlib import import_module

tok, model, ids, records = import_module("01_run_loop").run()
import_module("02_show_result").run(tok, ids)
import_module("03_step_timing").run(records)
import_module("04_length_sweep").run(model)
import_module("05_where_waste").run(records)
