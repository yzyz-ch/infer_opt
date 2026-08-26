"""按顺序跑完 00d。03 最慢（约 20～30 秒）。"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _path  # noqa: F401

from importlib import import_module

_tok, model, _ids, records = import_module("01_cached_loop").run()
import_module("02_prefill_vs_decode").run(model, records)
import_module("03_when_faster").run(model)
import_module("04_bf16_and_exercises").run()
