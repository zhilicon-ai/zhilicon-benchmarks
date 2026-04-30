"""Run a workload spec (YAML) through a tiny analytical performance model.

The runner deliberately does not import ``perf_models``; the perf-model
framework lives in a parallel package and may or may not be present.
Instead, this runner uses a small built-in roofline model that can be
swapped for the real one in production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ---- Workload spec ---------------------------------------------------------

@dataclass
class WorkloadKernel:
    """A single kernel in a workload spec."""

    name: str
    flops: float
    bytes_: int
    intensity: float = 0.0  # FLOPs per byte (computed if zero)
    iterations: int = 1

    def __post_init__(self) -> None:
        if self.flops <= 0:
            raise ValueError("flops must be positive")
        if self.bytes_ <= 0:
            raise ValueError("bytes_ must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.intensity == 0.0:
            self.intensity = self.flops / self.bytes_


@dataclass
class WorkloadSpec:
    """A YAML-loadable description of a workload."""

    name: str
    description: str
    kernels: List[WorkloadKernel] = field(default_factory=list)

    def total_flops(self) -> float:
        return sum(k.flops * k.iterations for k in self.kernels)

    def total_bytes(self) -> int:
        return sum(k.bytes_ * k.iterations for k in self.kernels)


def load_workload(path: str | Path) -> WorkloadSpec:
    """Load a workload spec from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workload spec must be a YAML mapping")
    return parse_workload(raw)


def parse_workload(data: Dict[str, Any]) -> WorkloadSpec:
    """Construct a ``WorkloadSpec`` from a plain dict."""
    if "name" not in data or "kernels" not in data:
        raise ValueError("workload spec must have 'name' and 'kernels'")
    kernels = []
    for entry in data["kernels"]:
        kernels.append(
            WorkloadKernel(
                name=entry["name"],
                flops=float(entry["flops"]),
                bytes_=int(entry["bytes"]),
                intensity=float(entry.get("intensity", 0.0)),
                iterations=int(entry.get("iterations", 1)),
            )
        )
    return WorkloadSpec(
        name=data["name"],
        description=data.get("description", ""),
        kernels=kernels,
    )


# ---- Tiny roofline model ---------------------------------------------------

@dataclass
class ChipModel:
    """Minimal compute/memory roofline parameters."""

    peak_tflops: float
    peak_bw_gbs: float

    def __post_init__(self) -> None:
        if self.peak_tflops <= 0:
            raise ValueError("peak_tflops must be positive")
        if self.peak_bw_gbs <= 0:
            raise ValueError("peak_bw_gbs must be positive")

    def kernel_seconds(self, kernel: WorkloadKernel) -> float:
        """Return the predicted runtime for one iteration of a kernel."""
        compute_s = kernel.flops / (self.peak_tflops * 1e12)
        memory_s = kernel.bytes_ / (self.peak_bw_gbs * 1e9)
        return max(compute_s, memory_s)


@dataclass
class RunResult:
    """Result of running a workload through a model."""

    workload: str
    chip_peak_tflops: float
    chip_peak_bw_gbs: float
    total_seconds: float
    kernel_seconds: Dict[str, float]
    achieved_tflops: float


def run_workload(spec: WorkloadSpec, chip: ChipModel) -> RunResult:
    """Roofline-evaluate a workload on a chip model."""
    per_kernel: Dict[str, float] = {}
    total_s = 0.0
    for k in spec.kernels:
        per_kernel[k.name] = chip.kernel_seconds(k) * k.iterations
        total_s += per_kernel[k.name]
    total_flops = spec.total_flops()
    achieved = (total_flops / total_s) / 1e12 if total_s > 0 else 0.0
    return RunResult(
        workload=spec.name,
        chip_peak_tflops=chip.peak_tflops,
        chip_peak_bw_gbs=chip.peak_bw_gbs,
        total_seconds=total_s,
        kernel_seconds=per_kernel,
        achieved_tflops=achieved,
    )
