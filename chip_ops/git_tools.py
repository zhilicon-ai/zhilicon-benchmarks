"""Git/GitHub helpers for the chip-ops program.

Pure analytical helpers + thin wrappers around ``git`` and ``gh``. The
network-touching wrappers accept a ``runner`` callable so they can be
unit-tested without invoking real subprocesses.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence


# Forbidden tokens (case-insensitive).
AI_REFERENCE_PATTERNS: tuple[str, ...] = (
    "claude",
    "anthropic",
    "copilot",
    "chatgpt",
    "openai",
)


# Type alias for a subprocess runner: takes argv, returns stdout string.
Runner = Callable[[Sequence[str]], str]


def _default_runner(argv: Sequence[str]) -> str:
    """Run ``argv`` and return stdout (raises ``CalledProcessError`` on fail)."""
    out = subprocess.run(
        list(argv), check=True, capture_output=True, text=True
    )
    return out.stdout


@dataclass
class PullRequest:
    """Lightweight PR record (subset of ``gh pr list`` fields)."""

    number: int
    title: str
    author: str
    state: str
    url: str = ""
    additions: int = 0
    deletions: int = 0


def list_prs(
    repo: str,
    author: Optional[str] = None,
    *,
    runner: Runner = _default_runner,
    state: str = "all",
) -> List[PullRequest]:
    """List PRs in ``repo`` (optionally filtered by ``author``)."""
    argv: list[str] = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        state,
        "--json",
        "number,title,author,state,url,additions,deletions",
        "--limit",
        "200",
    ]
    if author:
        argv.extend(["--author", author])
    raw = runner(argv)
    data = json.loads(raw or "[]")
    return [
        PullRequest(
            number=int(item.get("number", 0)),
            title=str(item.get("title", "")),
            author=str(
                item.get("author", {}).get("login", "")
                if isinstance(item.get("author"), dict)
                else item.get("author", "")
            ),
            state=str(item.get("state", "")),
            url=str(item.get("url", "")),
            additions=int(item.get("additions", 0) or 0),
            deletions=int(item.get("deletions", 0) or 0),
        )
        for item in data
    ]


def count_loc(paths: Iterable[str | Path]) -> int:
    """Total physical-line count across the given files (skips missing)."""
    total = 0
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        try:
            total += sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        except UnicodeDecodeError:
            # Binary file: skip silently.
            continue
    return total


@dataclass
class AIScanResult:
    """Result of an AI-reference scan over a tree."""

    files_scanned: int = 0
    matches: List[tuple[str, int, str, str]] = field(default_factory=list)
    # (path, line_no, token, snippet)

    @property
    def has_matches(self) -> bool:
        return bool(self.matches)


def scan_for_ai_references(
    root: str | Path,
    *,
    extensions: tuple[str, ...] = (".py", ".md", ".txt", ".yml", ".yaml"),
    patterns: tuple[str, ...] = AI_REFERENCE_PATTERNS,
) -> AIScanResult:
    """Walk ``root`` and find any forbidden AI references.

    Matches are returned as 4-tuples (path, line_no, token, snippet).
    """
    root_path = Path(root)
    result = AIScanResult()
    if not root_path.exists():
        return result
    pattern_re = re.compile(
        r"\b(" + "|".join(re.escape(p) for p in patterns) + r")\b",
        re.IGNORECASE,
    )
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        if path.suffix.lower() not in extensions:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        result.files_scanned += 1
        for line_no, line in enumerate(text.splitlines(), start=1):
            m = pattern_re.search(line)
            if m:
                result.matches.append(
                    (
                        str(path),
                        line_no,
                        m.group(1).lower(),
                        line.strip()[:200],
                    )
                )
    return result


def authors_for_paths(
    paths: Iterable[str | Path], *, runner: Runner = _default_runner
) -> List[str]:
    """Return unique git authors who touched ``paths`` (best-effort)."""
    authors: set[str] = set()
    for p in paths:
        argv = ["git", "log", "--format=%ae", "--", str(p)]
        try:
            out = runner(argv)
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            line = line.strip()
            if line:
                authors.add(line)
    return sorted(authors)
