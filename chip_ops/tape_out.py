"""Tape-out package builder.

Builds an immutable bundle of tape-out artifacts (GDS, LEF, libs, reports,
sign-off logs) and emits a SHA-256 manifest for downstream foundry hand-off.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TapeOutArtifact:
    """A single artifact in a tape-out bundle."""

    relative_path: str
    sha256: str
    bytes_: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes_,
        }


@dataclass
class TapeOutPackage:
    """A complete tape-out package + manifest."""

    chip_id: str
    revision: str
    artifacts: List[TapeOutArtifact] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def total_bytes(self) -> int:
        return sum(a.bytes_ for a in self.artifacts)

    def manifest_dict(self) -> Dict[str, Any]:
        return {
            "chip_id": self.chip_id,
            "revision": self.revision,
            "created_at": self.created_at,
            "artifact_count": len(self.artifacts),
            "total_bytes": self.total_bytes(),
            "artifacts": [a.to_dict() for a in self.artifacts],
        }

    def manifest_json(self) -> str:
        return json.dumps(self.manifest_dict(), indent=2, sort_keys=True)


def _hash_file(path: Path, *, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def build_package(
    *, chip_id: str, revision: str, source_dir: str | Path
) -> TapeOutPackage:
    """Walk ``source_dir`` and produce a deterministic tape-out package.

    Files are added in sorted order (so manifests are reproducible).
    Empty directories are silently skipped.
    """
    if not chip_id:
        raise ValueError("chip_id must not be empty")
    if not revision:
        raise ValueError("revision must not be empty")
    src = Path(source_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"source_dir not found or not a directory: {src}")

    artifacts: List[TapeOutArtifact] = []
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src).as_posix()
        artifacts.append(
            TapeOutArtifact(
                relative_path=rel,
                sha256=_hash_file(path),
                bytes_=path.stat().st_size,
            )
        )

    return TapeOutPackage(chip_id=chip_id, revision=revision, artifacts=artifacts)


def write_manifest(pkg: TapeOutPackage, manifest_path: str | Path) -> Path:
    """Persist the tape-out manifest to disk."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(pkg.manifest_json(), encoding="utf-8")
    return p


def verify_manifest(
    *, source_dir: str | Path, manifest_path: str | Path
) -> List[str]:
    """Re-hash files under ``source_dir`` and return the list of mismatched paths.

    A return value of ``[]`` means the bundle matches the manifest exactly.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    src = Path(source_dir)
    expected = {a["path"]: a["sha256"] for a in manifest["artifacts"]}
    actual: Dict[str, str] = {}
    for path in (p for p in src.rglob("*") if p.is_file()):
        actual[path.relative_to(src).as_posix()] = _hash_file(path)

    bad: List[str] = []
    for rel, exp_hash in expected.items():
        if actual.get(rel) != exp_hash:
            bad.append(rel)
    for rel in actual:
        if rel not in expected:
            bad.append(rel)
    return sorted(bad)
