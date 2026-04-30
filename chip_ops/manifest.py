"""Per-chip manifest: a typed dataclass with JSON load/save.

A ``ChipManifest`` is the canonical record describing a single chip
in the Zhilicon catalog. Manifests are persisted as JSON to keep them
git-diffable and tool-agnostic.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_VALID_NODES = {"3nm", "4nm", "5nm", "6nm", "7nm", "10nm", "12nm", "16nm", "28nm"}


@dataclass
class ChipManifest:
    """A canonical chip-record manifest.

    Parameters
    ----------
    chip_id:        unique short identifier (e.g. ``"zh-leo-a0"``).
    family:         architectural family the chip belongs to.
    process_node:   foundry process geometry (e.g. ``"5nm"``).
    die_area_mm2:   die area in square millimeters.
    transistor_count_billions: total transistor count in billions.
    target_tdp_w:   target thermal-design-power in watts.
    target_clock_ghz: nominal compute-clock target in GHz.
    memory_gb:      packaged HBM/DRAM in GB.
    interconnect:   chip-to-chip fabric (``"nvlink-style"`` etc.).
    owner:          GitHub handle of accountable PIC.
    tags:           free-form labels (e.g. ``["ai-inference", "edge"]``).
    created_at:     RFC-3339 timestamp; auto-populated when omitted.
    """

    chip_id: str
    family: str
    process_node: str
    die_area_mm2: float
    transistor_count_billions: float
    target_tdp_w: float
    target_clock_ghz: float
    memory_gb: int
    interconnect: str
    owner: str
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.chip_id:
            raise ValueError("chip_id must not be empty")
        if self.process_node not in _VALID_NODES:
            raise ValueError(
                f"process_node {self.process_node!r} not in {sorted(_VALID_NODES)}"
            )
        if self.die_area_mm2 <= 0:
            raise ValueError("die_area_mm2 must be positive")
        if self.transistor_count_billions <= 0:
            raise ValueError("transistor_count_billions must be positive")
        if self.target_tdp_w <= 0:
            raise ValueError("target_tdp_w must be positive")
        if self.target_clock_ghz <= 0:
            raise ValueError("target_clock_ghz must be positive")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ---- I/O ---------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for JSON."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        """Write the manifest to ``path`` (creates parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChipManifest":
        """Construct a manifest from a plain dictionary."""
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "ChipManifest":
        """Load a manifest from a JSON file."""
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(raw))


def aggregate_die_area(manifests: List[ChipManifest]) -> float:
    """Sum die area across a list of manifests."""
    return sum(m.die_area_mm2 for m in manifests)


def filter_by_node(
    manifests: List[ChipManifest], process_node: str
) -> List[ChipManifest]:
    """Return manifests matching a given process node."""
    return [m for m in manifests if m.process_node == process_node]
