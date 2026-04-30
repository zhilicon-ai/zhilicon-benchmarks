"""Tests for chip_ops.workload_runner."""
from __future__ import annotations

from pathlib import Path

import pytest

from chip_ops.workload_runner import (
    ChipModel,
    WorkloadKernel,
    WorkloadSpec,
    load_workload,
    parse_workload,
    run_workload,
)


def test_kernel_intensity_auto_computed() -> None:
    k = WorkloadKernel(name="gemm", flops=1e9, bytes_=int(1e6))
    assert k.intensity == pytest.approx(1e3)


def test_kernel_intensity_explicit() -> None:
    k = WorkloadKernel(name="gemm", flops=1e9, bytes_=int(1e6), intensity=5.0)
    assert k.intensity == 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flops": -1, "bytes_": 1},
        {"flops": 1, "bytes_": -1},
        {"flops": 1, "bytes_": 1, "iterations": 0},
    ],
)
def test_kernel_validation(kwargs: dict) -> None:
    base = {"name": "k", "flops": 1, "bytes_": 1}
    base.update(kwargs)
    with pytest.raises(ValueError):
        WorkloadKernel(**base)  # type: ignore[arg-type]


def test_workload_totals() -> None:
    spec = WorkloadSpec(
        name="bench",
        description="",
        kernels=[
            WorkloadKernel(name="a", flops=1e9, bytes_=int(1e6)),
            WorkloadKernel(name="b", flops=2e9, bytes_=int(2e6), iterations=2),
        ],
    )
    assert spec.total_flops() == pytest.approx(1e9 + 4e9)
    assert spec.total_bytes() == int(1e6 + 4e6)


def test_chip_model_validation() -> None:
    with pytest.raises(ValueError):
        ChipModel(peak_tflops=0, peak_bw_gbs=10)
    with pytest.raises(ValueError):
        ChipModel(peak_tflops=10, peak_bw_gbs=-1)


def test_run_workload_compute_bound() -> None:
    chip = ChipModel(peak_tflops=10.0, peak_bw_gbs=1.0)
    spec = WorkloadSpec(
        name="cb",
        description="compute-bound",
        kernels=[WorkloadKernel("k", flops=1e12, bytes_=10)],
    )
    result = run_workload(spec, chip)
    # 1 TFLOP at 10 TFLOPS peak = 0.1s
    assert result.total_seconds == pytest.approx(0.1, rel=0.01)
    assert result.achieved_tflops == pytest.approx(10.0, rel=0.01)


def test_run_workload_memory_bound() -> None:
    chip = ChipModel(peak_tflops=1000.0, peak_bw_gbs=10.0)
    spec = WorkloadSpec(
        name="mb",
        description="memory-bound",
        kernels=[WorkloadKernel("k", flops=1, bytes_=int(10e9))],
    )
    result = run_workload(spec, chip)
    # 10 GB at 10 GB/s = 1s
    assert result.total_seconds == pytest.approx(1.0, rel=0.01)


def test_parse_workload_minimum() -> None:
    data = {
        "name": "bench",
        "kernels": [{"name": "k", "flops": 1.0, "bytes": 1}],
    }
    spec = parse_workload(data)
    assert spec.name == "bench"
    assert len(spec.kernels) == 1


def test_parse_workload_missing_keys_raises() -> None:
    with pytest.raises(ValueError):
        parse_workload({"kernels": []})


def test_load_workload_yaml(tmp_path: Path) -> None:
    yaml_text = (
        "name: leo-bench\n"
        "description: leo benchmark\n"
        "kernels:\n"
        "  - name: gemm\n"
        "    flops: 1.0e9\n"
        "    bytes: 1000000\n"
        "    iterations: 4\n"
    )
    p = tmp_path / "wl.yaml"
    p.write_text(yaml_text)
    spec = load_workload(p)
    assert spec.name == "leo-bench"
    assert len(spec.kernels) == 1
    assert spec.kernels[0].iterations == 4


def test_load_workload_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- this is a list, not a mapping\n")
    with pytest.raises(ValueError):
        load_workload(p)


def test_run_workload_kernel_seconds_recorded() -> None:
    chip = ChipModel(peak_tflops=10.0, peak_bw_gbs=10.0)
    spec = WorkloadSpec(
        name="x",
        description="",
        kernels=[
            WorkloadKernel("a", flops=1e10, bytes_=int(1e8)),
            WorkloadKernel("b", flops=1e10, bytes_=int(1e8)),
        ],
    )
    result = run_workload(spec, chip)
    assert set(result.kernel_seconds.keys()) == {"a", "b"}
    assert all(v > 0 for v in result.kernel_seconds.values())
