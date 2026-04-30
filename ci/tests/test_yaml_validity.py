"""
Validates every YAML file under the repo's ci/ tree.

Each file is parsed via PyYAML's safe_load. Workflow files are additionally
checked for required top-level keys (`name`, `on`, `jobs` for direct workflows
or `workflow_call`/`workflow_dispatch` for templates).
"""

from __future__ import annotations

import pathlib
from typing import List

import pytest
import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _yaml_files_under(root: pathlib.Path) -> List[pathlib.Path]:
    if not root.exists():
        return []
    return sorted(
        list(root.rglob("*.yml")) + list(root.rglob("*.yaml"))
    )


def _ci_yaml_files() -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for sub in ["ci"]:
        files.extend(_yaml_files_under(REPO_ROOT / sub))
    return files


def test_at_least_one_yaml_found():
    """Sanity: the repo must have at least one YAML to validate."""
    files = _ci_yaml_files()
    assert len(files) > 0, "no YAML files found"


@pytest.mark.parametrize("path", _ci_yaml_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_yaml_parses(path: pathlib.Path):
    """Every YAML file must parse without error."""
    with path.open() as fh:
        try:
            yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            pytest.fail(f"{path.relative_to(REPO_ROOT)}: {exc}")


def _is_workflow_file(path: pathlib.Path) -> bool:
    """Workflow files live under a `workflows/` directory."""
    return "workflows" in path.parts


@pytest.mark.parametrize(
    "path",
    [p for p in _ci_yaml_files() if _is_workflow_file(p)],
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_workflow_has_required_keys(path: pathlib.Path):
    """Every workflow must have `name`, `on`, and `jobs`."""
    with path.open() as fh:
        doc = yaml.safe_load(fh)
    assert isinstance(doc, dict), f"{path}: top-level not a mapping"
    # YAML's `on` is special (parsed as bool True in safe_load), so accept either
    on_key = "on" if "on" in doc else (True if True in doc else None)
    assert on_key is not None, f"{path}: missing `on:` trigger"
    assert "name" in doc, f"{path}: missing `name:`"
    assert "jobs" in doc, f"{path}: missing `jobs:`"
    assert isinstance(doc["jobs"], dict), f"{path}: jobs not a mapping"
    assert len(doc["jobs"]) > 0, f"{path}: jobs is empty"


@pytest.mark.parametrize(
    "path",
    [p for p in _ci_yaml_files() if _is_workflow_file(p)],
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_workflow_jobs_have_runs_on_or_uses(path: pathlib.Path):
    """Each job must declare runs-on (direct) or uses (reusable workflow call)."""
    with path.open() as fh:
        doc = yaml.safe_load(fh)
    for job_name, job in doc.get("jobs", {}).items():
        assert isinstance(job, dict), f"{path}: job {job_name} is not a mapping"
        if "uses" in job:
            assert isinstance(job["uses"], str)
        else:
            assert "runs-on" in job, (
                f"{path}: job {job_name} missing both `runs-on:` and `uses:`"
            )


def test_no_hardcoded_secrets_in_workflow():
    """Sanity: no obvious AWS/GH PAT/SSH keys hard-coded into workflows."""
    bad_patterns = ["AKIA", "ghp_", "BEGIN RSA PRIVATE KEY"]
    for path in _ci_yaml_files():
        text = path.read_text()
        for pattern in bad_patterns:
            assert pattern not in text, (
                f"{path}: potential hardcoded secret matching '{pattern}'"
            )
