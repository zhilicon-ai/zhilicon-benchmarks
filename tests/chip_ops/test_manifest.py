"""Tests for chip_ops.manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chip_ops.manifest import (
    ChipManifest,
    aggregate_die_area,
    filter_by_node,
)


def test_basic_construction(sample_manifest: ChipManifest) -> None:
    assert sample_manifest.chip_id == "zh-leo-a0"
    assert sample_manifest.process_node == "5nm"
    assert sample_manifest.created_at is not None


def test_to_dict_roundtrip(sample_manifest: ChipManifest) -> None:
    d = sample_manifest.to_dict()
    rebuilt = ChipManifest.from_dict(d)
    assert rebuilt == sample_manifest


def test_save_load(tmp_path: Path, sample_manifest: ChipManifest) -> None:
    p = tmp_path / "nested" / "manifest.json"
    sample_manifest.save(p)
    loaded = ChipManifest.load(p)
    assert loaded.chip_id == sample_manifest.chip_id
    assert loaded.tags == ["ai-inference", "edge"]


def test_to_json_is_pretty(sample_manifest: ChipManifest) -> None:
    text = sample_manifest.to_json()
    parsed = json.loads(text)
    assert parsed["chip_id"] == "zh-leo-a0"
    # Sorted keys means chip_id comes before family in the JSON text.
    assert text.index('"chip_id"') < text.index('"family"')


@pytest.mark.parametrize(
    "field,value",
    [
        ("chip_id", ""),
        ("die_area_mm2", -1),
        ("die_area_mm2", 0),
        ("transistor_count_billions", -3.0),
        ("target_tdp_w", 0),
        ("target_clock_ghz", -0.5),
        ("memory_gb", 0),
    ],
)
def test_invalid_values_raise(field: str, value: object) -> None:
    base = dict(
        chip_id="x",
        family="leo",
        process_node="5nm",
        die_area_mm2=10.0,
        transistor_count_billions=1.0,
        target_tdp_w=10.0,
        target_clock_ghz=1.0,
        memory_gb=8,
        interconnect="pci",
        owner="r",
    )
    base[field] = value
    with pytest.raises(ValueError):
        ChipManifest(**base)  # type: ignore[arg-type]


def test_invalid_node_raises() -> None:
    with pytest.raises(ValueError, match="process_node"):
        ChipManifest(
            chip_id="x",
            family="leo",
            process_node="2nm",  # not in valid set
            die_area_mm2=10.0,
            transistor_count_billions=1.0,
            target_tdp_w=10.0,
            target_clock_ghz=1.0,
            memory_gb=8,
            interconnect="pci",
            owner="r",
        )


def test_aggregate_die_area(sample_manifest: ChipManifest) -> None:
    second = ChipManifest(
        chip_id="zh-virgo-a0",
        family="virgo",
        process_node="7nm",
        die_area_mm2=100.0,
        transistor_count_billions=20.0,
        target_tdp_w=80.0,
        target_clock_ghz=1.4,
        memory_gb=16,
        interconnect="pcie",
        owner="r",
    )
    assert aggregate_die_area([sample_manifest, second]) == pytest.approx(512.0)


def test_filter_by_node(sample_manifest: ChipManifest) -> None:
    second = ChipManifest(
        chip_id="zh-virgo-a0",
        family="virgo",
        process_node="7nm",
        die_area_mm2=100.0,
        transistor_count_billions=20.0,
        target_tdp_w=80.0,
        target_clock_ghz=1.4,
        memory_gb=16,
        interconnect="pcie",
        owner="r",
    )
    only_5nm = filter_by_node([sample_manifest, second], "5nm")
    assert only_5nm == [sample_manifest]
    only_7nm = filter_by_node([sample_manifest, second], "7nm")
    assert only_7nm == [second]


def test_explicit_created_at_is_preserved() -> None:
    m = ChipManifest(
        chip_id="x",
        family="leo",
        process_node="5nm",
        die_area_mm2=10.0,
        transistor_count_billions=1.0,
        target_tdp_w=10.0,
        target_clock_ghz=1.0,
        memory_gb=8,
        interconnect="pci",
        owner="r",
        created_at="2024-01-01T00:00:00+00:00",
    )
    assert m.created_at == "2024-01-01T00:00:00+00:00"


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ChipManifest.load(tmp_path / "missing.json")
