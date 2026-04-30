"""Tests for chip_ops.git_tools."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Sequence

import pytest

from chip_ops.git_tools import (
    AI_REFERENCE_PATTERNS,
    AIScanResult,
    PullRequest,
    authors_for_paths,
    count_loc,
    list_prs,
    scan_for_ai_references,
)


def test_count_loc_basic(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("line1\nline2\nline3\n")
    b = tmp_path / "b.py"
    b.write_text("only-one\n")
    assert count_loc([a, b]) == 4


def test_count_loc_skips_missing(tmp_path: Path) -> None:
    a = tmp_path / "exists.py"
    a.write_text("line\n")
    assert count_loc([a, tmp_path / "missing.py"]) == 1


def test_count_loc_skips_binary(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("ok\n")
    binp = tmp_path / "b.bin"
    binp.write_bytes(b"\x00\x01\xff\xfe\x80\x90")
    assert count_loc([a, binp]) == 1


def test_scan_finds_match(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("# fine code\n")
    (tmp_path / "bad.md").write_text("Powered by Claude here\n")
    result = scan_for_ai_references(tmp_path)
    assert result.has_matches
    assert result.files_scanned == 2
    paths = [m[0] for m in result.matches]
    assert any("bad.md" in p for p in paths)


def test_scan_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("# clean\n")
    result = scan_for_ai_references(tmp_path)
    assert not result.has_matches


def test_scan_missing_root() -> None:
    result = scan_for_ai_references("/nope/does-not-exist-xyz")
    assert result == AIScanResult()


def test_scan_skips_unknown_extensions(tmp_path: Path) -> None:
    (tmp_path / "bin.dat").write_text("openai everywhere\n")
    result = scan_for_ai_references(tmp_path)
    assert not result.has_matches


def test_scan_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("# AnTHrOpIc reference\n")
    result = scan_for_ai_references(tmp_path)
    assert result.has_matches
    assert result.matches[0][2] == "anthropic"


def test_pattern_set() -> None:
    expected = {"claude", "anthropic", "copilot", "chatgpt", "openai"}
    assert set(AI_REFERENCE_PATTERNS) == expected


def test_list_prs_with_fake_runner() -> None:
    fake_data = json.dumps(
        [
            {
                "number": 7,
                "title": "feat: foo",
                "author": {"login": "ramesh"},
                "state": "OPEN",
                "url": "https://example/7",
                "additions": 10,
                "deletions": 3,
            }
        ]
    )
    captured: List[Sequence[str]] = []

    def fake(argv: Sequence[str]) -> str:
        captured.append(argv)
        return fake_data

    prs = list_prs("zh/zhilicon-benchmarks", "ramesh", runner=fake)
    assert len(prs) == 1
    assert prs[0].number == 7
    assert prs[0].author == "ramesh"
    # Sanity: --author flag included.
    assert "--author" in captured[0]


def test_list_prs_without_author() -> None:
    captured: List[Sequence[str]] = []

    def fake(argv: Sequence[str]) -> str:
        captured.append(argv)
        return "[]"

    prs = list_prs("z/x", runner=fake)
    assert prs == []
    assert "--author" not in captured[0]


def test_pull_request_dataclass() -> None:
    pr = PullRequest(number=1, title="t", author="a", state="OPEN")
    assert pr.additions == 0


def test_authors_for_paths_with_fake_runner(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("hi\n")

    def fake(_argv: Sequence[str]) -> str:
        return "alice@example.com\nbob@example.com\n"

    authors = authors_for_paths([f], runner=fake)
    assert authors == ["alice@example.com", "bob@example.com"]


def test_authors_for_paths_handles_called_process_error(
    tmp_path: Path,
) -> None:
    f = tmp_path / "x.py"
    f.write_text("hi\n")

    def fake(_argv: Sequence[str]) -> str:
        raise subprocess.CalledProcessError(1, list(_argv))

    assert authors_for_paths([f], runner=fake) == []


def test_scan_collects_line_numbers(tmp_path: Path) -> None:
    p = tmp_path / "f.md"
    p.write_text("clean\nclean\nopenai is here\n")
    result = scan_for_ai_references(tmp_path)
    assert result.matches[0][1] == 3
