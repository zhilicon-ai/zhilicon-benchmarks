"""
Base classes for chip-level performance models.

All chip models inherit from `ChipModel`, which captures the analytical
parameters needed to project performance, power, and area for a given
workload mix.

The unit conventions are documented in `perf_models/__init__.py`. The
classes here are intentionally pure-Python with no external deps so the
framework can be vendored into vendor lab environments without bringing
NumPy/SciPy along.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PrecisionMode(Enum):
    """Numeric precision modes supported by the compute fabric.

    The energy-per-operation table varies across precision modes; an INT8
    MAC is roughly 4x cheaper than an FP16 MAC, which in turn is ~3x
    cheaper than an FP32 MAC. Microscaling (MX) formats fall in between.
    """

    FP4_E2M1 = "fp4_e2m1"      # Microscaling FP4 (Blackwell-class)
    FP8_E4M3 = "fp8_e4m3"      # H100/H200-class FP8 (per-tensor scale)
    FP8_E5M2 = "fp8_e5m2"      # FP8 alt (wider dynamic range)
    BF16     = "bf16"
    FP16     = "fp16"
    TF32     = "tf32"          # 19-bit mantissa training format
    FP32     = "fp32"
    FP64     = "fp64"
    INT4     = "int4"
    INT8     = "int8"
    INT16    = "int16"
    INT32    = "int32"


# Per-MAC energy in picojoules (pJ), measured at 0.75 V nominal.
# These values come from published TSMC N5 / N3 numbers, scaled for the
# Zhilicon process variants. See `docs/perf-model-validation.md` for the
# derivation chain.
ENERGY_PER_MAC_PJ: Dict[PrecisionMode, float] = {
    PrecisionMode.FP4_E2M1: 0.030,
    PrecisionMode.FP8_E4M3: 0.060,
    PrecisionMode.FP8_E5M2: 0.060,
    PrecisionMode.BF16:     0.180,
    PrecisionMode.FP16:     0.180,
    PrecisionMode.TF32:     0.350,
    PrecisionMode.FP32:     0.500,
    PrecisionMode.FP64:     1.200,
    PrecisionMode.INT4:     0.018,
    PrecisionMode.INT8:     0.045,
    PrecisionMode.INT16:    0.150,
    PrecisionMode.INT32:    0.380,
}


@dataclass(frozen=True)
class ProcessNode:
    """Foundry process node parameters.

    The Zhilicon portfolio targets multiple process nodes:

    * Sentinel-1: TSMC N3
    * Horizon-1:  GlobalFoundries 22FDX SOI (rad-hard process)
    * Discovery-1: TSMC N3
    * Nexus-1:   TSMC N3
    * Prometheus: TSMC N3 (eventually N2)
    """

    name: str
    feature_size_nm: float            # Marketing node name (e.g., 3.0)
    nominal_voltage_mv: int           # Nominal supply at fast corner
    threshold_voltage_mv: int         # NMOS Vth at typ corner
    transistor_density_mtx_mm2: float # Million transistors per mm^2
    sram_bitcell_area_um2: float      # 6T SRAM bitcell area
    leakage_per_mtx_uw: float         # Leakage at 25 deg C, typ
    metal_stack_layers: int           # BEOL metal layer count


# Built-in process node catalog
PROCESS_NODES: Dict[str, ProcessNode] = {
    "tsmc_n3": ProcessNode(
        name="tsmc_n3",
        feature_size_nm=3.0,
        nominal_voltage_mv=750,
        threshold_voltage_mv=350,
        transistor_density_mtx_mm2=292,
        sram_bitcell_area_um2=0.021,
        leakage_per_mtx_uw=0.00012,
        metal_stack_layers=18,
    ),
    "tsmc_n5": ProcessNode(
        name="tsmc_n5",
        feature_size_nm=5.0,
        nominal_voltage_mv=750,
        threshold_voltage_mv=380,
        transistor_density_mtx_mm2=171,
        sram_bitcell_area_um2=0.0210,
        leakage_per_mtx_uw=0.00014,
        metal_stack_layers=15,
    ),
    "gf22fdx_soi": ProcessNode(
        name="gf22fdx_soi",
        feature_size_nm=22.0,
        nominal_voltage_mv=800,
        threshold_voltage_mv=420,
        transistor_density_mtx_mm2=24,     # Lower, but RHBD cells
        sram_bitcell_area_um2=0.092,
        leakage_per_mtx_uw=0.00006,        # SOI = lower leakage
        metal_stack_layers=11,
    ),
    "tsmc_n2": ProcessNode(
        name="tsmc_n2",
        feature_size_nm=2.0,
        nominal_voltage_mv=700,
        threshold_voltage_mv=320,
        transistor_density_mtx_mm2=420,
        sram_bitcell_area_um2=0.0198,
        leakage_per_mtx_uw=0.00018,
        metal_stack_layers=20,
    ),
}


@dataclass(frozen=True)
class OperatingPoint:
    """A DVFS operating point: (voltage, frequency) tuple plus derived
    parameters used to project dynamic / leakage power.

    The 5-chip portfolio uses 4-8 named operating points per chip; the
    runtime selects between them via the DVFS controller (see
    `rtl/ip/power/zhi_dvfs_controller.sv`).
    """

    name: str
    voltage_mv: int
    frequency_ghz: float
    activity_factor: float = 0.30   # Fraction of MACs active per cycle
    temperature_c: float = 50.0     # Junction temperature

    @property
    def voltage_v(self) -> float:
        """Voltage in volts."""
        return self.voltage_mv / 1000.0

    @property
    def frequency_hz(self) -> float:
        """Frequency in Hz."""
        return self.frequency_ghz * 1e9


@dataclass
class ComputeFabric:
    """Description of a chip's compute fabric.

    A fabric is parameterised by:
    * The total MAC count (sum across all PEs)
    * The supported precision modes
    * The peak frequency of the fabric
    * The energy-per-MAC table (defaults to global ENERGY_PER_MAC_PJ)
    """

    total_macs: int                            # Total MAC units
    supported_precisions: List[PrecisionMode]  # What this fabric supports
    peak_freq_ghz: float                       # Peak fabric clock
    energy_pj_per_mac: Optional[Dict[PrecisionMode, float]] = None

    def get_energy_pj(self, prec: PrecisionMode) -> float:
        """Return per-MAC energy at this precision."""
        if self.energy_pj_per_mac and prec in self.energy_pj_per_mac:
            return self.energy_pj_per_mac[prec]
        return ENERGY_PER_MAC_PJ[prec]

    def peak_throughput_tops(self, prec: PrecisionMode) -> float:
        """Peak throughput in TOPS (Tera-Operations Per Second).

        Each MAC counts as 2 ops (multiply + add).
        """
        return (self.total_macs * 2 * self.peak_freq_ghz) / 1000.0

    def peak_throughput_tflops(self, prec: PrecisionMode) -> float:
        """Peak throughput in TFLOPS for floating-point precision."""
        if prec not in (PrecisionMode.FP4_E2M1, PrecisionMode.FP8_E4M3,
                        PrecisionMode.FP8_E5M2, PrecisionMode.BF16,
                        PrecisionMode.FP16, PrecisionMode.TF32,
                        PrecisionMode.FP32, PrecisionMode.FP64):
            raise ValueError(f"{prec} is not a floating-point format")
        return self.peak_throughput_tops(prec)


@dataclass
class MemoryHierarchy:
    """Description of a chip's memory hierarchy.

    Covers L1 / L2 / L3 caches, scratchpad SRAM, on-package HBM/LPDDR,
    and any far-memory tiers. Each level captures capacity, line size,
    bandwidth, and read/write energy per access.
    """

    l1_kb: int
    l1_bw_gb_s: float
    l1_read_energy_pj: float
    l1_write_energy_pj: float

    l2_kb: int
    l2_bw_gb_s: float
    l2_read_energy_pj: float
    l2_write_energy_pj: float

    l3_kb: int
    l3_bw_gb_s: float
    l3_read_energy_pj: float
    l3_write_energy_pj: float

    hbm_capacity_gb: float
    hbm_bw_gb_s: float
    hbm_read_energy_pj: float
    hbm_write_energy_pj: float


@dataclass
class InterconnectFabric:
    """Description of a chip's on-die interconnect.

    Captures NoC topology, link width, frequency, and energy per flit
    transfer. Used to model communication-bound kernels.
    """

    topology: str                  # "mesh", "torus", "dragonfly", etc.
    rows: int
    cols: int
    link_bits: int                 # Bits per link
    link_freq_ghz: float           # Link clock
    flit_size_bytes: int
    energy_per_flit_pj: float
    hop_latency_cycles: int = 1

    @property
    def num_routers(self) -> int:
        return self.rows * self.cols

    def link_bandwidth_gb_s(self) -> float:
        """Single-link bandwidth in GB/s."""
        return (self.link_bits * self.link_freq_ghz) / 8.0

    def bisection_bandwidth_gb_s(self) -> float:
        """Bisection bandwidth -- the key metric for all-to-all traffic."""
        # For a 2D mesh, bisection bandwidth = min(rows, cols) * link_bw
        if self.topology == "mesh":
            return min(self.rows, self.cols) * self.link_bandwidth_gb_s()
        # For 2D torus, double (wrap-around links)
        if self.topology == "torus":
            return 2 * min(self.rows, self.cols) * self.link_bandwidth_gb_s()
        # For dragonfly, use a coarse approximation
        if self.topology == "dragonfly":
            return self.num_routers // 4 * self.link_bandwidth_gb_s()
        raise ValueError(f"Unknown topology: {self.topology}")


@dataclass
class ChipModel:
    """Top-level chip performance model.

    All 5 Zhilicon chips inherit from this base class. Subclasses fill in
    chip-specific parameters and add chip-specific kernel coverage.

    Methods provide:
    * `peak_compute(prec)` -- peak throughput in TOPS at this precision
    * `kernel_runtime(kernel)` -- analytical runtime for a kernel
    * `kernel_energy(kernel)` -- analytical energy for a kernel
    * `total_power_w(workload)` -- total power for a workload
    """

    name: str
    process: ProcessNode
    die_area_mm2: float
    package_tdp_w: float
    operating_points: List[OperatingPoint]
    fabric: ComputeFabric
    memory: MemoryHierarchy
    interconnect: InterconnectFabric

    def __post_init__(self) -> None:
        # Validate configuration
        if self.die_area_mm2 <= 0:
            raise ValueError(f"die_area_mm2 must be > 0, got {self.die_area_mm2}")
        if self.package_tdp_w <= 0:
            raise ValueError(f"package_tdp_w must be > 0, got {self.package_tdp_w}")
        if not self.operating_points:
            raise ValueError("must provide at least one operating point")
        # Sort operating points by frequency for convenience
        self.operating_points.sort(key=lambda op: op.frequency_ghz)

    @property
    def lowest_power_op(self) -> OperatingPoint:
        """Lowest power (Vmin / Fmin) operating point."""
        return self.operating_points[0]

    @property
    def highest_perf_op(self) -> OperatingPoint:
        """Highest performance (Vmax / Fmax) operating point."""
        return self.operating_points[-1]

    @property
    def nominal_op(self) -> OperatingPoint:
        """Nominal (mid-range) operating point."""
        return self.operating_points[len(self.operating_points) // 2]

    def total_transistor_count_mtx(self) -> float:
        """Total transistor count, in millions."""
        return self.die_area_mm2 * self.process.transistor_density_mtx_mm2

    def total_leakage_w(self, op: OperatingPoint) -> float:
        """Static (leakage) power at a given operating point.

        Leakage scales roughly exponentially with voltage (0.25 V swing
        => ~10x leakage change) and with temperature (10 deg C => ~2x
        leakage change). The scaling factor coefficients here come from
        TSMC N3/N5 SPICE-derived empirical fits.
        """
        nominal_leakage_uw = (
            self.total_transistor_count_mtx() * self.process.leakage_per_mtx_uw * 1e6
        )
        # Voltage scaling: I_leak ~ V^a, a ~= 4 for FinFET
        v_scale = (op.voltage_mv / self.process.nominal_voltage_mv) ** 4.0
        # Temperature scaling: doubles every 10 deg C above 25 C
        t_scale = 2.0 ** ((op.temperature_c - 25.0) / 10.0)
        return nominal_leakage_uw * v_scale * t_scale * 1e-6  # Convert to W

    def dynamic_power_w(self, op: OperatingPoint, prec: PrecisionMode) -> float:
        """Dynamic power at a given operating point + precision.

        Pdyn = activity * C_eff * V^2 * F
        We approximate C_eff via the per-MAC energy table.
        """
        energy_pj = self.fabric.get_energy_pj(prec)
        # Voltage scaling: dyn power ~ V^2
        v_scale = (op.voltage_mv / self.process.nominal_voltage_mv) ** 2.0
        # Active MACs per second
        active_macs = self.fabric.total_macs * op.activity_factor
        # Effective ops/cycle = MACs * 2 (mult + add)
        # Energy per op-cycle (pJ) at this voltage
        energy_pj_op_v = energy_pj * v_scale
        # Total dynamic power
        return active_macs * op.frequency_hz * energy_pj_op_v * 1e-12

    def total_power_w(self, op: OperatingPoint, prec: PrecisionMode) -> float:
        """Total chip power = dynamic + leakage."""
        return self.dynamic_power_w(op, prec) + self.total_leakage_w(op)

    def perf_per_watt_tops_w(
        self, op: OperatingPoint, prec: PrecisionMode
    ) -> float:
        """Perf-per-watt (TOPS / W)."""
        peak = self.fabric.peak_throughput_tops(prec)
        # Scale peak by frequency ratio (op might be running at lower freq)
        scaled_peak = peak * (op.frequency_ghz / self.fabric.peak_freq_ghz)
        achievable = scaled_peak * op.activity_factor
        power = self.total_power_w(op, prec)
        return achievable / power if power > 0 else 0.0

    def serializable(self) -> Dict:
        """Convert this model to a JSON-serializable dict."""
        return {
            "name": self.name,
            "process": dataclasses.asdict(self.process),
            "die_area_mm2": self.die_area_mm2,
            "package_tdp_w": self.package_tdp_w,
            "operating_points": [dataclasses.asdict(op) for op in self.operating_points],
            "fabric": {
                "total_macs": self.fabric.total_macs,
                "peak_freq_ghz": self.fabric.peak_freq_ghz,
                "supported_precisions": [p.value for p in self.fabric.supported_precisions],
            },
            "memory": dataclasses.asdict(self.memory),
            "interconnect": dataclasses.asdict(self.interconnect),
        }

    def summary_table(self) -> str:
        """Pretty-print a summary table of the chip's PPA."""
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"  {self.name} -- chip performance model")
        lines.append(f"{'='*70}")
        lines.append(f"  Process:           {self.process.name} ({self.process.feature_size_nm} nm)")
        lines.append(f"  Die area:          {self.die_area_mm2:.0f} mm^2")
        lines.append(f"  Transistor count:  {self.total_transistor_count_mtx():.0f} M")
        lines.append(f"  TDP:               {self.package_tdp_w:.0f} W")
        lines.append(f"  Operating points:  {len(self.operating_points)}")
        lines.append(f"")
        lines.append(f"  Compute fabric:")
        lines.append(f"    Total MACs:      {self.fabric.total_macs:,}")
        lines.append(f"    Peak frequency:  {self.fabric.peak_freq_ghz:.2f} GHz")
        lines.append(f"    Precisions:      {', '.join(p.value for p in self.fabric.supported_precisions)}")
        lines.append(f"")
        lines.append(f"  Memory hierarchy:")
        lines.append(f"    L1:   {self.memory.l1_kb:>6} KB,  {self.memory.l1_bw_gb_s:>6.1f} GB/s")
        lines.append(f"    L2:   {self.memory.l2_kb:>6} KB,  {self.memory.l2_bw_gb_s:>6.1f} GB/s")
        lines.append(f"    L3:   {self.memory.l3_kb:>6} KB,  {self.memory.l3_bw_gb_s:>6.1f} GB/s")
        lines.append(f"    HBM:  {self.memory.hbm_capacity_gb:>5.1f} GB,  {self.memory.hbm_bw_gb_s:>6.1f} GB/s")
        lines.append(f"")
        lines.append(f"  Interconnect: {self.interconnect.topology} {self.interconnect.rows}x{self.interconnect.cols}")
        lines.append(f"    Bisection BW: {self.interconnect.bisection_bandwidth_gb_s():,.1f} GB/s")
        lines.append(f"")
        # Per-op PPA
        lines.append(f"  PPA at each operating point:")
        lines.append(f"    {'Name':<12} {'V':>6} {'F':>8} {'Power':>9} {'TOPS/W':>10}")
        for op in self.operating_points:
            try:
                fp16_pw = self.perf_per_watt_tops_w(op, PrecisionMode.FP16)
            except (ValueError, KeyError):
                fp16_pw = 0.0
            lines.append(
                f"    {op.name:<12} {op.voltage_v:>5.2f}V {op.frequency_ghz:>6.2f}GHz "
                f"{self.total_power_w(op, PrecisionMode.FP16):>7.1f}W {fp16_pw:>8.1f}"
            )
        lines.append(f"{'='*70}")
        return "\n".join(lines)
