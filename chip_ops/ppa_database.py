"""SQLite-backed historical PPA (perf/power/area) tracker."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass
class PPARecord:
    """A single PPA datapoint for a chip revision."""

    chip_id: str
    revision: str
    perf_score: float
    power_w: float
    area_mm2: float
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.perf_score < 0:
            raise ValueError("perf_score must be non-negative")
        if self.power_w < 0:
            raise ValueError("power_w must be non-negative")
        if self.area_mm2 <= 0:
            raise ValueError("area_mm2 must be positive")
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )


class PPADatabase:
    """SQLite store for PPA records with simple analytics."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS ppa (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chip_id TEXT NOT NULL,
        revision TEXT NOT NULL,
        perf_score REAL NOT NULL,
        power_w REAL NOT NULL,
        area_mm2 REAL NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chip ON ppa(chip_id);
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._cursor() as cur:
            cur.executescript(self.SCHEMA)

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        finally:
            cur.close()

    def insert(self, record: PPARecord) -> int:
        """Insert a record; return its row id."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO ppa
                  (chip_id, revision, perf_score, power_w, area_mm2, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.chip_id,
                    record.revision,
                    record.perf_score,
                    record.power_w,
                    record.area_mm2,
                    record.timestamp,
                ),
            )
            return int(cur.lastrowid or 0)

    def fetch_chip(self, chip_id: str) -> list[PPARecord]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT chip_id, revision, perf_score, power_w, area_mm2, "
                "timestamp FROM ppa WHERE chip_id = ? ORDER BY id",
                (chip_id,),
            )
            rows = cur.fetchall()
        return [PPARecord(**dict(row)) for row in rows]

    def best_perf_per_watt(self, chip_id: str) -> PPARecord | None:
        """Return revision with highest perf/watt for ``chip_id`` (or None)."""
        records = self.fetch_chip(chip_id)
        if not records:
            return None
        return max(records, key=lambda r: r.perf_score / max(r.power_w, 1e-9))

    def revision_count(self, chip_id: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM ppa WHERE chip_id = ?",
                (chip_id,),
            )
            row = cur.fetchone()
        return int(row["n"])

    def all_chip_ids(self) -> list[str]:
        with self._cursor() as cur:
            cur.execute("SELECT DISTINCT chip_id FROM ppa ORDER BY chip_id")
            rows = cur.fetchall()
        return [row["chip_id"] for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PPADatabase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
