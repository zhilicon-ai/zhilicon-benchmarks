"""Tests for chip_ops.ate_program."""
from __future__ import annotations

from pathlib import Path

import pytest

from chip_ops.ate_program import (
    GOOD_BINS,
    SCRAP_BINS,
    ATEProgram,
    BinResult,
    YieldTracker,
    classify,
)


def _program() -> ATEProgram:
    return ATEProgram(
        program_id="prog-leo-a0",
        chip_id="zh-leo-a0",
        test_list=["scan", "vmin", "fmax"],
        bin_map={"scan": 1, "vmin": 2, "fmax": 5},
    )


def test_program_construction_ok() -> None:
    prog = _program()
    assert prog.program_id == "prog-leo-a0"
    assert prog.target_yield_pct == 80.0


def test_program_save_load_round_trip(tmp_path: Path) -> None:
    prog = _program()
    p = tmp_path / "prog.json"
    prog.save(p)
    loaded = ATEProgram.load(p)
    assert loaded.program_id == prog.program_id
    assert loaded.bin_map == prog.bin_map


def test_program_empty_test_list_raises() -> None:
    with pytest.raises(ValueError):
        ATEProgram(
            program_id="x",
            chip_id="x",
            test_list=[],
            bin_map={},
        )


def test_program_missing_bin_map_entry_raises() -> None:
    with pytest.raises(ValueError):
        ATEProgram(
            program_id="x",
            chip_id="x",
            test_list=["scan", "fmax"],
            bin_map={"scan": 1},
        )


def test_program_invalid_target_yield_raises() -> None:
    with pytest.raises(ValueError):
        ATEProgram(
            program_id="x",
            chip_id="x",
            test_list=["scan"],
            bin_map={"scan": 1},
            target_yield_pct=120.0,
        )


def test_yield_tracker_basic() -> None:
    yt = YieldTracker(program_id="p")
    yt.add("d1", 1)
    yt.add("d2", 2)
    yt.add("d3", 99)
    yt.add("d4", 5)
    assert yt.total == 4
    assert yt.good_count == 2
    assert yt.scrap_count == 1
    assert yt.yield_pct() == pytest.approx(50.0)


def test_yield_tracker_meets_target() -> None:
    yt = YieldTracker(program_id="p")
    yt.add_many([("d1", 1), ("d2", 1), ("d3", 5)])
    assert yt.yield_pct() == pytest.approx(200.0 / 3)
    assert not yt.meets_target(80.0)
    assert yt.meets_target(50.0)


def test_yield_tracker_empty() -> None:
    yt = YieldTracker(program_id="p")
    assert yt.total == 0
    assert yt.yield_pct() == 0.0


def test_bin_distribution() -> None:
    yt = YieldTracker(program_id="p")
    yt.add_many([("d1", 1), ("d2", 1), ("d3", 5), ("d4", 99)])
    dist = yt.bin_distribution()
    assert dist == {1: 2, 5: 1, 99: 1}


def test_classify() -> None:
    assert classify(1) == "good"
    assert classify(2) == "good"
    assert classify(99) == "scrap"
    assert classify(7) == "reject"


def test_bin_result_dataclass() -> None:
    r = BinResult("d1", 1)
    assert r.die_id == "d1"
    assert r.bin_number == 1


def test_constants() -> None:
    assert GOOD_BINS == frozenset({1, 2})
    assert SCRAP_BINS == frozenset({99})
