"""Macht die Projektmodule importierbar, egal von wo `pytest` gestartet wird."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
