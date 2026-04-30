"""Shared test fixtures for chip_ops tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from chip_ops.manifest import ChipManifest


@pytest.fixture
def sample_manifest() -> ChipManifest:
    return ChipManifest(
        chip_id="zh-leo-a0",
        family="leo",
        process_node="5nm",
        die_area_mm2=412.0,
        transistor_count_billions=85.5,
        target_tdp_w=350.0,
        target_clock_ghz=1.85,
        memory_gb=80,
        interconnect="nvlink-style",
        owner="ramesh",
        tags=["ai-inference", "edge"],
    )


@pytest.fixture
def tape_out_tree(tmp_path: Path) -> Path:
    """Return a populated tape-out source directory."""
    src = tmp_path / "tape_out_src"
    src.mkdir()
    (src / "design.gds").write_bytes(b"GDS_BINARY_DATA")
    (src / "macro.lef").write_text("VERSION 5.7 ;\n")
    sub = src / "reports"
    sub.mkdir()
    (sub / "drc.log").write_text("DRC PASS\n")
    (sub / "lvs.log").write_text("LVS PASS\n")
    return src
