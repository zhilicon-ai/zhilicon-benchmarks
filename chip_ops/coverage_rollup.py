"""Aggregate Verilator + UVM coverage across multiple chips/runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class CoverageReport:
    """A coverage report from a simulator run."""

    chip_id: str
    source: str  # "verilator" or "uvm" typically
    line_pct: float
    branch_pct: float
    toggle_pct: float
    fsm_pct: float
    extra: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for fld in ("line_pct", "branch_pct", "toggle_pct", "fsm_pct"):
            value = getattr(self, fld)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{fld} out of range: {value}")

    def overall(self) -> float:
        """Equally-weighted blend of the 4 main coverage axes."""
        return (
            self.line_pct + self.branch_pct + self.toggle_pct + self.fsm_pct
        ) / 4.0


def merge_reports(reports: Iterable[CoverageReport]) -> dict[str, float]:
    """Return averaged coverage across reports for the same chip group.

    Output keys: ``line_pct``, ``branch_pct``, ``toggle_pct``, ``fsm_pct``,
    ``overall_pct``. Raises if no reports are provided.
    """
    items = list(reports)
    if not items:
        raise ValueError("at least one CoverageReport required")
    n = len(items)
    return {
        "line_pct": sum(r.line_pct for r in items) / n,
        "branch_pct": sum(r.branch_pct for r in items) / n,
        "toggle_pct": sum(r.toggle_pct for r in items) / n,
        "fsm_pct": sum(r.fsm_pct for r in items) / n,
        "overall_pct": sum(r.overall() for r in items) / n,
    }


def rollup_by_chip(
    reports: Iterable[CoverageReport],
) -> dict[str, dict[str, float]]:
    """Group reports by chip_id then merge each group."""
    grouped: dict[str, list[CoverageReport]] = {}
    for r in reports:
        grouped.setdefault(r.chip_id, []).append(r)
    return {chip: merge_reports(items) for chip, items in grouped.items()}


def coverage_gaps(
    summary: dict[str, float], threshold: float = 90.0
) -> list[str]:
    """List axes whose coverage falls below ``threshold``."""
    return [
        axis
        for axis in ("line_pct", "branch_pct", "toggle_pct", "fsm_pct")
        if summary.get(axis, 0.0) < threshold
    ]
