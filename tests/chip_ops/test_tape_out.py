"""Tests for chip_ops.tape_out."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chip_ops.tape_out import (
    TapeOutArtifact,
    TapeOutPackage,
    build_package,
    verify_manifest,
    write_manifest,
)


def test_build_package_basic(tape_out_tree: Path) -> None:
    pkg = build_package(chip_id="zh-x", revision="A0", source_dir=tape_out_tree)
    names = [a.relative_path for a in pkg.artifacts]
    assert "design.gds" in names
    assert "reports/drc.log" in names
    assert pkg.total_bytes() > 0


def test_artifacts_are_sorted(tape_out_tree: Path) -> None:
    pkg = build_package(chip_id="zh-x", revision="A0", source_dir=tape_out_tree)
    names = [a.relative_path for a in pkg.artifacts]
    assert names == sorted(names)


def test_manifest_round_trip(tape_out_tree: Path, tmp_path: Path) -> None:
    pkg = build_package(chip_id="zh-x", revision="A0", source_dir=tape_out_tree)
    out = tmp_path / "manifest.json"
    write_manifest(pkg, out)
    data = json.loads(out.read_text())
    assert data["chip_id"] == "zh-x"
    assert data["revision"] == "A0"
    assert data["artifact_count"] == len(pkg.artifacts)


def test_verify_manifest_clean(tape_out_tree: Path, tmp_path: Path) -> None:
    pkg = build_package(chip_id="zh-x", revision="A0", source_dir=tape_out_tree)
    manifest = tmp_path / "m.json"
    write_manifest(pkg, manifest)
    assert verify_manifest(source_dir=tape_out_tree, manifest_path=manifest) == []


def test_verify_manifest_detects_change(
    tape_out_tree: Path, tmp_path: Path
) -> None:
    pkg = build_package(chip_id="zh-x", revision="A0", source_dir=tape_out_tree)
    manifest = tmp_path / "m.json"
    write_manifest(pkg, manifest)
    # Mutate one of the files.
    (tape_out_tree / "design.gds").write_bytes(b"DIFFERENT")
    bad = verify_manifest(source_dir=tape_out_tree, manifest_path=manifest)
    assert "design.gds" in bad


def test_verify_manifest_detects_extra_file(
    tape_out_tree: Path, tmp_path: Path
) -> None:
    pkg = build_package(chip_id="zh-x", revision="A0", source_dir=tape_out_tree)
    manifest = tmp_path / "m.json"
    write_manifest(pkg, manifest)
    (tape_out_tree / "rogue.txt").write_text("extra")
    bad = verify_manifest(source_dir=tape_out_tree, manifest_path=manifest)
    assert "rogue.txt" in bad


def test_empty_chip_id_raises(tape_out_tree: Path) -> None:
    with pytest.raises(ValueError):
        build_package(chip_id="", revision="A0", source_dir=tape_out_tree)


def test_empty_revision_raises(tape_out_tree: Path) -> None:
    with pytest.raises(ValueError):
        build_package(chip_id="zh-x", revision="", source_dir=tape_out_tree)


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_package(chip_id="zh-x", revision="A0", source_dir=tmp_path / "nope")


def test_artifact_to_dict() -> None:
    a = TapeOutArtifact("foo.gds", "deadbeef" * 8, 1024)
    d = a.to_dict()
    assert d == {"path": "foo.gds", "sha256": "deadbeef" * 8, "bytes": 1024}


def test_manifest_dict_keys() -> None:
    pkg = TapeOutPackage(chip_id="zh-x", revision="A0")
    keys = set(pkg.manifest_dict().keys())
    assert keys == {
        "chip_id",
        "revision",
        "created_at",
        "artifact_count",
        "total_bytes",
        "artifacts",
    }


def test_total_bytes_zero_when_empty() -> None:
    pkg = TapeOutPackage(chip_id="zh-x", revision="A0")
    assert pkg.total_bytes() == 0
