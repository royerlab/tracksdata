"""ASV benchmark package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
