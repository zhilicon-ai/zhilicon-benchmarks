"""Command-line driver for chip_ops (``python -m chip_ops``)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .ate_program import YieldTracker, classify
from .coverage_rollup import CoverageReport, merge_reports
from .git_tools import scan_for_ai_references
from .lifecycle import STAGES, ChipLifecycle
from .manifest import ChipManifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chip_ops", description="Chip-ops CLI")
    p.add_argument("--version", action="version", version=f"chip_ops {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- manifest dump
    mdump = sub.add_parser("manifest-dump", help="Dump a manifest JSON file")
    mdump.add_argument("path", type=Path)

    # ---- lifecycle stages
    sub.add_parser("lifecycle-stages", help="List the canonical lifecycle stages")

    # ---- yield from CSV-like input
    yld = sub.add_parser("yield", help="Compute yield from die,bin pairs (stdin)")
    yld.add_argument("--target", type=float, default=80.0)

    # ---- coverage average
    cov = sub.add_parser(
        "coverage-summary",
        help="Compute average coverage across one or more JSON reports",
    )
    cov.add_argument("paths", nargs="+", type=Path)

    # ---- ai scan
    scan = sub.add_parser(
        "scan-ai", help="Scan a tree for forbidden AI references"
    )
    scan.add_argument("root", type=Path)

    # ---- progress for one chip
    prog = sub.add_parser(
        "lifecycle-progress",
        help="Compute progress percentage given a current stage",
    )
    prog.add_argument("stage", choices=STAGES)

    return p


def _cmd_manifest_dump(args: argparse.Namespace) -> int:
    m = ChipManifest.load(args.path)
    print(m.to_json())
    return 0


def _cmd_lifecycle_stages(_args: argparse.Namespace) -> int:
    for s in STAGES:
        print(s)
    return 0


def _cmd_yield(args: argparse.Namespace) -> int:
    tracker = YieldTracker(program_id="cli")
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        die_id, bin_str = (s.strip() for s in line.split(",", 1))
        tracker.add(die_id, int(bin_str))
    summary = {
        "total": tracker.total,
        "good": tracker.good_count,
        "scrap": tracker.scrap_count,
        "yield_pct": round(tracker.yield_pct(), 4),
        "meets_target": tracker.meets_target(args.target),
        "classification_examples": {
            "1": classify(1),
            "5": classify(5),
            "99": classify(99),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if tracker.meets_target(args.target) else 1


def _cmd_coverage_summary(args: argparse.Namespace) -> int:
    reports: List[CoverageReport] = []
    for path in args.paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        reports.append(
            CoverageReport(
                chip_id=data["chip_id"],
                source=data.get("source", "unknown"),
                line_pct=float(data["line_pct"]),
                branch_pct=float(data["branch_pct"]),
                toggle_pct=float(data["toggle_pct"]),
                fsm_pct=float(data["fsm_pct"]),
            )
        )
    summary = merge_reports(reports)
    print(json.dumps({k: round(v, 4) for k, v in summary.items()}, indent=2))
    return 0


def _cmd_scan_ai(args: argparse.Namespace) -> int:
    result = scan_for_ai_references(args.root)
    payload = {
        "files_scanned": result.files_scanned,
        "match_count": len(result.matches),
        "matches": [
            {"path": p, "line": ln, "token": tok, "snippet": snip}
            for (p, ln, tok, snip) in result.matches
        ],
    }
    print(json.dumps(payload, indent=2))
    return 1 if result.has_matches else 0


def _cmd_lifecycle_progress(args: argparse.Namespace) -> int:
    chip = ChipLifecycle("cli", initial_stage=args.stage)
    print(json.dumps({"stage": chip.stage, "progress_pct": chip.progress_pct()}))
    return 0


_COMMANDS = {
    "manifest-dump": _cmd_manifest_dump,
    "lifecycle-stages": _cmd_lifecycle_stages,
    "yield": _cmd_yield,
    "coverage-summary": _cmd_coverage_summary,
    "scan-ai": _cmd_scan_ai,
    "lifecycle-progress": _cmd_lifecycle_progress,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = _COMMANDS[args.command]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - covered via tests
    raise SystemExit(main())
