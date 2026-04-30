"""Repo-root pytest conftest.

Adds the repository root to ``sys.path`` so that flat layout packages
(``perf_models``, ``chip_ops``, etc.) import cleanly during ``pytest`` runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
