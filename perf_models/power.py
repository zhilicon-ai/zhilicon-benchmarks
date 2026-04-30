"""Power-decomposition utilities.

Functions in this module take a `ChipModel` + `OperatingPoint` and return
detailed power breakdowns: dynamic vs leakage, compute vs memory vs
interconnect, on-die vs off-die. Used for power-budget what-if analysis
and to validate post-silicon measurement against the analytical model.

All energies in pJ, all powers in W.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .base import ChipModel, OperatingPoint, PrecisionMode


@dataclass
class PowerBreakdown:
    """Detailed power breakdown at a single operating point."""

    chip_name: str
    op_name: str

    # Active categories
    compute_dynamic_w: float = 0.0
    memory_dynamic_w: float = 0.0
    interconnect_dynamic_w: float = 0.0
    io_dynamic_w: float = 0.0      # PCIe/Eth/UCIe SerDes power

    # Leakage (always-on)
    leakage_w: float = 0.0

    # Total
    total_w: float = 0.0

    @property
    def dynamic_w(self) -> float:
        return (
            self.compute_dynamic_w
            + self.memory_dynamic_w
            + self.interconnect_dynamic_w
            + self.io_dynamic_w
        )

    @property
    def dynamic_fraction(self) -> float:
        if self.total_w == 0:
            return 0.0
        return self.dynamic_w / self.total_w

    @property
    def leakage_fraction(self) -> float:
        if self.total_w == 0:
            return 0.0
        return self.leakage_w / self.total_w

    def as_dict(self) -> Dict[str, float]:
        return {
            "chip": self.chip_name,
            "op":   self.op_name,
            "compute_dynamic_w":      self.compute_dynamic_w,
            "memory_dynamic_w":       self.memory_dynamic_w,
            "interconnect_dynamic_w": self.interconnect_dynamic_w,
            "io_dynamic_w":           self.io_dynamic_w,
            "leakage_w":              self.leakage_w,
            "dynamic_total_w":        self.dynamic_w,
            "total_w":                self.total_w,
            "dynamic_fraction":       self.dynamic_fraction,
            "leakage_fraction":       self.leakage_fraction,
        }

    def __str__(self) -> str:
        lines = []
        lines.append(f"=== Power breakdown for {self.chip_name} @ {self.op_name} ===")
        lines.append(f"  Compute dyn:      {self.compute_dynamic_w:>7.2f} W "
                     f"({(self.compute_dynamic_w/self.total_w*100) if self.total_w else 0:>5.1f} %)")
        lines.append(f"  Memory dyn:       {self.memory_dynamic_w:>7.2f} W "
                     f"({(self.memory_dynamic_w/self.total_w*100) if self.total_w else 0:>5.1f} %)")
        lines.append(f"  Interconnect dyn: {self.interconnect_dynamic_w:>7.2f} W "
                     f"({(self.interconnect_dynamic_w/self.total_w*100) if self.total_w else 0:>5.1f} %)")
        lines.append(f"  I/O dyn:          {self.io_dynamic_w:>7.2f} W "
                     f"({(self.io_dynamic_w/self.total_w*100) if self.total_w else 0:>5.1f} %)")
        lines.append(f"  Leakage:          {self.leakage_w:>7.2f} W "
                     f"({self.leakage_fraction*100:>5.1f} %)")
        lines.append(f"  ----------------------------------------------------")
        lines.append(f"  TOTAL:            {self.total_w:>7.2f} W")
        return "\n".join(lines)


def compute_power_breakdown(
    chip: ChipModel,
    op: OperatingPoint,
    prec: PrecisionMode = PrecisionMode.FP16,
    memory_activity: float = 0.30,
    interconnect_activity: float = 0.20,
    io_activity: float = 0.10,
) -> PowerBreakdown:
    """Project a detailed power breakdown for a chip at an operating point.

    Args:
        chip: ChipModel
        op: OperatingPoint
        prec: precision used for compute power
        memory_activity: fraction of HBM traffic active (0..1)
        interconnect_activity: NoC link activity (0..1)
        io_activity: SerDes I/O activity (0..1)
    """
    breakdown = PowerBreakdown(chip_name=chip.name, op_name=op.name)

    # Compute dynamic power -- delegates to ChipModel.dynamic_power_w
    breakdown.compute_dynamic_w = chip.dynamic_power_w(op, prec)

    # Memory dynamic power: estimate from HBM bandwidth and energy/byte
    if memory_activity > 0:
        # 1 GB/s of HBM at hbm_read_energy_pj per 64B line
        # bytes/s = bw_gb_s * 1e9
        # accesses/s = bytes/s / 64
        # power = accesses/s * energy_pj * 1e-12
        active_bw_bytes_per_s = chip.memory.hbm_bw_gb_s * 1e9 * memory_activity
        accesses_per_s = active_bw_bytes_per_s / 64.0
        avg_energy = (chip.memory.hbm_read_energy_pj + chip.memory.hbm_write_energy_pj) / 2
        breakdown.memory_dynamic_w = accesses_per_s * avg_energy * 1e-12

    # Interconnect dynamic power: estimate from flit traffic
    if interconnect_activity > 0:
        ic = chip.interconnect
        # Each link transfers (link_freq_ghz * 1e9) flits per second at peak
        flits_per_s_per_link = ic.link_freq_ghz * 1e9
        total_links = ic.num_routers * 4  # Approximate: 4 links per router
        active_flits_per_s = total_links * flits_per_s_per_link * interconnect_activity
        breakdown.interconnect_dynamic_w = (
            active_flits_per_s * ic.energy_per_flit_pj * 1e-12
        )

    # I/O dynamic power: rough estimate -- assume 5W of PCIe/Ethernet/UCIe
    # at full activity, scaled by activity
    nominal_io_w = 5.0
    breakdown.io_dynamic_w = nominal_io_w * io_activity

    # Leakage
    breakdown.leakage_w = chip.total_leakage_w(op)

    breakdown.total_w = (
        breakdown.compute_dynamic_w
        + breakdown.memory_dynamic_w
        + breakdown.interconnect_dynamic_w
        + breakdown.io_dynamic_w
        + breakdown.leakage_w
    )
    return breakdown


def power_sweep(
    chip: ChipModel,
    prec: PrecisionMode = PrecisionMode.FP16,
) -> List[PowerBreakdown]:
    """Sweep across all operating points and return a list of breakdowns."""
    return [compute_power_breakdown(chip, op, prec) for op in chip.operating_points]


def find_optimal_op_for_perf_per_watt(
    chip: ChipModel,
    prec: PrecisionMode = PrecisionMode.FP16,
) -> OperatingPoint:
    """Find the operating point with the best TOPS/W (energy efficiency)."""
    best_op = chip.lowest_power_op
    best_pw = chip.perf_per_watt_tops_w(best_op, prec)
    for op in chip.operating_points:
        pw = chip.perf_per_watt_tops_w(op, prec)
        if pw > best_pw:
            best_op = op
            best_pw = pw
    return best_op


def power_within_tdp(
    chip: ChipModel,
    op: OperatingPoint,
    prec: PrecisionMode = PrecisionMode.FP16,
    margin: float = 0.05,
) -> bool:
    """Check whether projected power fits within TDP with a margin.

    Args:
        margin: safety margin as fraction (default 5%).
    Returns:
        True iff total_w * (1 + margin) <= package_tdp_w
    """
    bd = compute_power_breakdown(chip, op, prec)
    return bd.total_w * (1.0 + margin) <= chip.package_tdp_w
