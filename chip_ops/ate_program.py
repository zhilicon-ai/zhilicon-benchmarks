"""ATE (automated test equipment) program loader, bin policy, yield tracker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Standard ATE bin numbering convention used in this project.
GOOD_BINS: frozenset[int] = frozenset({1, 2})
SCRAP_BINS: frozenset[int] = frozenset({99})


@dataclass
class ATEProgram:
    """An ATE program definition: tests + pass/fail mappings to bins."""

    program_id: str
    chip_id: str
    test_list: list[str]
    bin_map: dict[str, int]
    target_yield_pct: float = 80.0

    def __post_init__(self) -> None:
        if not self.test_list:
            raise ValueError("test_list must contain at least one test")
        missing = [t for t in self.test_list if t not in self.bin_map]
        if missing:
            raise ValueError(f"bin_map missing entries for tests: {missing}")
        if not 0.0 <= self.target_yield_pct <= 100.0:
            raise ValueError("target_yield_pct must be in [0,100]")

    @classmethod
    def load(cls, path: str | Path) -> "ATEProgram":
        data = json.loads(Path(path).read_text())
        return cls(
            program_id=data["program_id"],
            chip_id=data["chip_id"],
            test_list=list(data["test_list"]),
            bin_map={str(k): int(v) for k, v in data["bin_map"].items()},
            target_yield_pct=float(data.get("target_yield_pct", 80.0)),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "program_id": self.program_id,
                    "chip_id": self.chip_id,
                    "test_list": self.test_list,
                    "bin_map": self.bin_map,
                    "target_yield_pct": self.target_yield_pct,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return target


@dataclass
class BinResult:
    """A binned die from a single ATE run."""

    die_id: str
    bin_number: int


@dataclass
class YieldTracker:
    """Aggregate bin counts and yield computation."""

    program_id: str
    results: list[BinResult] = field(default_factory=list)

    def add(self, die_id: str, bin_number: int) -> None:
        self.results.append(BinResult(die_id=die_id, bin_number=bin_number))

    def add_many(self, items: Iterable[tuple[str, int]]) -> None:
        for die_id, bin_number in items:
            self.add(die_id, bin_number)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def good_count(self) -> int:
        return sum(1 for r in self.results if r.bin_number in GOOD_BINS)

    @property
    def scrap_count(self) -> int:
        return sum(1 for r in self.results if r.bin_number in SCRAP_BINS)

    def yield_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.good_count / self.total

    def bin_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for r in self.results:
            dist[r.bin_number] = dist.get(r.bin_number, 0) + 1
        return dist

    def meets_target(self, target_pct: float) -> bool:
        return self.yield_pct() >= target_pct


def classify(bin_number: int) -> str:
    """Classify a bin number as 'good', 'scrap', or 'reject'."""
    if bin_number in GOOD_BINS:
        return "good"
    if bin_number in SCRAP_BINS:
        return "scrap"
    return "reject"
