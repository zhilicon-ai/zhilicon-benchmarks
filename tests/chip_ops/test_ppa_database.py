"""Tests for chip_ops.ppa_database."""
from __future__ import annotations

from pathlib import Path

import pytest

from chip_ops.ppa_database import PPADatabase, PPARecord


def test_record_validation() -> None:
    with pytest.raises(ValueError):
        PPARecord(chip_id="x", revision="A0", perf_score=-1, power_w=10, area_mm2=100)
    with pytest.raises(ValueError):
        PPARecord(chip_id="x", revision="A0", perf_score=10, power_w=-1, area_mm2=100)
    with pytest.raises(ValueError):
        PPARecord(chip_id="x", revision="A0", perf_score=10, power_w=10, area_mm2=0)


def test_timestamp_auto_filled() -> None:
    r = PPARecord("x", "A0", 100, 10, 100)
    assert r.timestamp


def test_explicit_timestamp_preserved() -> None:
    r = PPARecord("x", "A0", 100, 10, 100, "2024-01-01")
    assert r.timestamp == "2024-01-01"


def test_insert_and_fetch_in_memory() -> None:
    db = PPADatabase()
    rid = db.insert(PPARecord("zh-leo", "A0", 100, 10, 200))
    assert rid >= 1
    rows = db.fetch_chip("zh-leo")
    assert len(rows) == 1
    assert rows[0].chip_id == "zh-leo"
    db.close()


def test_insert_multiple() -> None:
    with PPADatabase() as db:
        for r in (
            PPARecord("zh-leo", "A0", 100, 10, 200),
            PPARecord("zh-leo", "A1", 150, 11, 200),
            PPARecord("zh-leo", "A2", 180, 12, 200),
        ):
            db.insert(r)
        assert db.revision_count("zh-leo") == 3


def test_persisted_db(tmp_path: Path) -> None:
    db_path = tmp_path / "ppa.sqlite"
    with PPADatabase(db_path) as db:
        db.insert(PPARecord("zh-leo", "A0", 100, 10, 200))
    # Reopen.
    with PPADatabase(db_path) as db2:
        assert db2.revision_count("zh-leo") == 1


def test_best_perf_per_watt() -> None:
    with PPADatabase() as db:
        db.insert(PPARecord("zh-leo", "A0", 100, 10, 200))   # 10/W
        db.insert(PPARecord("zh-leo", "A1", 200, 10, 200))   # 20/W (winner)
        db.insert(PPARecord("zh-leo", "A2", 300, 30, 200))   # 10/W
        best = db.best_perf_per_watt("zh-leo")
        assert best is not None
        assert best.revision == "A1"


def test_best_perf_per_watt_empty() -> None:
    with PPADatabase() as db:
        assert db.best_perf_per_watt("nope") is None


def test_all_chip_ids() -> None:
    with PPADatabase() as db:
        db.insert(PPARecord("a", "A0", 100, 10, 200))
        db.insert(PPARecord("b", "A0", 100, 10, 200))
        db.insert(PPARecord("a", "A1", 100, 10, 200))
        assert db.all_chip_ids() == ["a", "b"]


def test_revision_count_zero_for_unknown() -> None:
    with PPADatabase() as db:
        assert db.revision_count("nope") == 0


def test_insert_returns_increasing_ids() -> None:
    with PPADatabase() as db:
        a = db.insert(PPARecord("x", "A0", 1, 1, 1))
        b = db.insert(PPARecord("x", "A1", 1, 1, 1))
        assert b > a


def test_fetch_orders_by_id() -> None:
    with PPADatabase() as db:
        db.insert(PPARecord("x", "A0", 1, 1, 1, "2024-01-01"))
        db.insert(PPARecord("x", "A1", 1, 1, 1, "2024-01-02"))
        rows = db.fetch_chip("x")
        assert [r.revision for r in rows] == ["A0", "A1"]
