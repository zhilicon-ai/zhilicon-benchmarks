"""Tests for chip_ops.cli."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from chip_ops.cli import main
from chip_ops.manifest import ChipManifest


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "chip_ops" in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "chip_ops" in out


def test_lifecycle_stages(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["lifecycle-stages"])
    assert rc == 0
    out = capsys.readouterr().out
    for s in ("design", "verify", "sim", "silicon", "pilot", "production"):
        assert s in out


def test_lifecycle_progress(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["lifecycle-progress", "sim"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "sim"
    assert payload["progress_pct"] == pytest.approx(40.0)


def test_manifest_dump(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    sample_manifest: ChipManifest,
) -> None:
    p = tmp_path / "m.json"
    sample_manifest.save(p)
    rc = main(["manifest-dump", str(p)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chip_id"] == "zh-leo-a0"


def test_yield_meets_target(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "# header\n"
            "d1,1\n"
            "d2,1\n"
            "d3,1\n"
            "d4,5\n"
        ),
    )
    rc = main(["yield", "--target", "70"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["good"] == 3
    assert payload["meets_target"] is True


def test_yield_misses_target(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("d1,5\nd2,99\n")
    )
    rc = main(["yield", "--target", "80"])
    assert rc == 1


def test_coverage_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(
            {
                "chip_id": "x",
                "source": "verilator",
                "line_pct": 80,
                "branch_pct": 80,
                "toggle_pct": 80,
                "fsm_pct": 80,
            }
        )
    )
    b.write_text(
        json.dumps(
            {
                "chip_id": "x",
                "source": "uvm",
                "line_pct": 100,
                "branch_pct": 100,
                "toggle_pct": 100,
                "fsm_pct": 100,
            }
        )
    )
    rc = main(["coverage-summary", str(a), str(b)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_pct"] == 90.0


def test_scan_ai_clean_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "ok.py").write_text("# clean code\n")
    rc = main(["scan-ai", str(tmp_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["match_count"] == 0


def test_scan_ai_dirty_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bad.py").write_text("# import openai\n")
    rc = main(["scan-ai", str(tmp_path)])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["match_count"] >= 1


def test_no_subcommand_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main([])


def test_main_module_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``python -m chip_ops`` import path executes."""
    import runpy

    monkeypatch.setattr("sys.argv", ["chip_ops", "lifecycle-stages"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("chip_ops", run_name="__main__")
    assert exc.value.code == 0
