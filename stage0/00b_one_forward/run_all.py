"""按顺序跑完 00b。在 stage0 目录下: python 00b_one_forward/run_all.py"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _path  # noqa: F401

from importlib import import_module

tok, model = import_module("01_load").run()
input_ids = import_module("02_tokenize").run(tok, model)
logits = import_module("03_one_forward").run(model, input_ids)
import_module("04_last_position").run(tok, model, input_ids, logits)
next_id = import_module("05_next_token").run(tok, logits)
import_module("06_summary").run(tok, input_ids, next_id)
