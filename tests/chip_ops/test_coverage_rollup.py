"""Tests for chip_ops.coverage_rollup."""
from __future__ import annotations

import pytest

from chip_ops.coverage_rollup import (
    CoverageReport,
    coverage_gaps,
    merge_reports,
    rollup_by_chip,
)


def _report(
    chip_id: str = "zh-x",
    source: str = "verilator",
    line: float = 95.0,
    branch: float = 90.0,
    toggle: float = 85.0,
    fsm: float = 100.0,
) -> CoverageReport:
    return CoverageReport(
        chip_id=chip_id,
        source=source,
        line_pct=line,
        branch_pct=branch,
        toggle_pct=toggle,
        fsm_pct=fsm,
    )


def test_overall() -> None:
    r = _report()
    assert r.overall() == pytest.approx((95 + 90 + 85 + 100) / 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("line_pct", -1),
        ("branch_pct", 101),
        ("toggle_pct", 200),
        ("fsm_pct", -10),
    ],
)
def test_invalid_pct_raises(field: str, value: float) -> None:
    base = dict(line=95.0, branch=90.0, toggle=85.0, fsm=100.0)
    base[field.split("_")[0]] = value
    with pytest.raises(ValueError):
        _report(**base)  # type: ignore[arg-type]


def test_merge_single_report() -> None:
    summary = merge_reports([_report()])
    assert summary["line_pct"] == 95.0
    assert "overall_pct" in summary


def test_merge_multiple_reports() -> None:
    a = _report(line=80, branch=80, toggle=80, fsm=80)
    b = _report(line=90, branch=90, toggle=90, fsm=90)
    summary = merge_reports([a, b])
    assert summary["line_pct"] == pytest.approx(85.0)
    assert summary["overall_pct"] == pytest.approx(85.0)


def test_merge_empty_raises() -> None:
    with pytest.raises(ValueError):
        merge_reports([])


def test_rollup_by_chip_groups() -> None:
    reports = [
        _report(chip_id="A", line=80),
        _report(chip_id="A", line=100),
        _report(chip_id="B", line=70),
    ]
    rolled = rollup_by_chip(reports)
    assert set(rolled.keys()) == {"A", "B"}
    assert rolled["A"]["line_pct"] == pytest.approx(90.0)
    assert rolled["B"]["line_pct"] == pytest.approx(70.0)


def test_coverage_gaps_below_threshold() -> None:
    summary = {
        "line_pct": 80.0,
        "branch_pct": 95.0,
        "toggle_pct": 70.0,
        "fsm_pct": 99.0,
    }
    gaps = coverage_gaps(summary, threshold=90.0)
    assert set(gaps) == {"line_pct", "toggle_pct"}


def test_coverage_gaps_all_clear() -> None:
    summary = {
        "line_pct": 95.0,
        "branch_pct": 95.0,
        "toggle_pct": 95.0,
        "fsm_pct": 95.0,
    }
    assert coverage_gaps(summary, threshold=90.0) == []


def test_coverage_gaps_default_threshold() -> None:
    summary = {
        "line_pct": 89.0,
        "branch_pct": 95.0,
        "toggle_pct": 95.0,
        "fsm_pct": 95.0,
    }
    assert coverage_gaps(summary) == ["line_pct"]


def test_extra_field_default() -> None:
    r = _report()
    assert r.extra == {}


def test_extra_field_round_trip() -> None:
    r = CoverageReport(
        chip_id="x",
        source="verilator",
        line_pct=95.0,
        branch_pct=95.0,
        toggle_pct=95.0,
        fsm_pct=95.0,
        extra={"assertion_pct": 80.0},
    )
    assert r.extra["assertion_pct"] == 80.0
