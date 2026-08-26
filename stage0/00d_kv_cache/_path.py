"""把 stage0 加进 sys.path，以便 import common。"""

import sys
from pathlib import Path

STAGE0 = Path(__file__).resolve().parent.parent
if str(STAGE0) not in sys.path:
    sys.path.insert(0, str(STAGE0))
