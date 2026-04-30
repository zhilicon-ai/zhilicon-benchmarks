"""chip_ops: Production tooling for the Zhilicon chip program.

Modules
-------
manifest         : per-chip manifest dataclass with JSON load/save
lifecycle        : finite-state machine governing chip lifecycle stages
tape_out         : tape-out package builder with SHA-256 manifest
ate_program      : ATE program loader, bin policy, and yield tracker
coverage_rollup  : aggregate Verilator + UVM coverage across chips
ppa_database     : SQLite-backed historical PPA tracker
workload_runner  : execute YAML workload specs against a perf model
mailbox          : in-process inter-team messaging stub
git_tools        : helpers for PR listing, LoC counting, AI-reference scanning
cli              : ``python -m chip_ops`` driver

The package targets Python 3.10+ and is fully type-hinted. It depends only on
the standard library plus PyYAML (for workload specs).
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "ate_program",
    "coverage_rollup",
    "git_tools",
    "lifecycle",
    "mailbox",
    "manifest",
    "ppa_database",
    "tape_out",
    "workload_runner",
]
