"""
Per-chip performance models for the 5-chip Zhilicon portfolio.

Each chip subclass populates the `ChipModel` base with concrete process,
fabric, memory, and interconnect parameters. Numbers come from the
authoritative architecture documents in `zhilicon-architecture/<chip>/`
and are validated by the test suite in `tests/test_chips.py`.

References:
* Sentinel-1: ZH-S1-ARCH-001 V4.1
* Horizon-1:  ZH-H1-UPG-001 V3.0-UPGRADE
* Discovery-1: ZH-D1-IMPL-001 V1
* Nexus-1:    ZH-N1-1T-001
* Prometheus: ZH-P1-V1-001
"""

from __future__ import annotations

from .base import (
    ChipModel,
    ComputeFabric,
    InterconnectFabric,
    MemoryHierarchy,
    OperatingPoint,
    PrecisionMode,
    PROCESS_NODES,
)


def Sentinel1Model() -> ChipModel:
    """Sentinel-1: crypto + PQC accelerator on TSMC N3, 82W TDP, 112 mm^2."""
    return ChipModel(
        name="Sentinel-1",
        process=PROCESS_NODES["tsmc_n3"],
        die_area_mm2=112.0,
        package_tdp_w=82.0,
        operating_points=[
            OperatingPoint("low",     voltage_mv=600, frequency_ghz=1.50, activity_factor=0.20),
            OperatingPoint("nominal", voltage_mv=750, frequency_ghz=2.50, activity_factor=0.30),
            OperatingPoint("turbo",   voltage_mv=850, frequency_ghz=3.20, activity_factor=0.40),
        ],
        fabric=ComputeFabric(
            total_macs=4096,                            # 16x crypto engines, 256 MACs each
            supported_precisions=[
                PrecisionMode.INT8, PrecisionMode.INT32, PrecisionMode.FP16,
            ],
            peak_freq_ghz=3.20,
        ),
        memory=MemoryHierarchy(
            l1_kb=64,    l1_bw_gb_s=1280.0, l1_read_energy_pj=8.0,  l1_write_energy_pj=10.0,
            l2_kb=512,   l2_bw_gb_s=400.0,  l2_read_energy_pj=22.0, l2_write_energy_pj=28.0,
            l3_kb=8192,  l3_bw_gb_s=200.0,  l3_read_energy_pj=80.0, l3_write_energy_pj=100.0,
            hbm_capacity_gb=24.0, hbm_bw_gb_s=900.0,
            hbm_read_energy_pj=400.0, hbm_write_energy_pj=480.0,
        ),
        interconnect=InterconnectFabric(
            topology="mesh", rows=4, cols=4,
            link_bits=128, link_freq_ghz=1.5,
            flit_size_bytes=16, energy_per_flit_pj=12.0,
            hop_latency_cycles=1,
        ),
    )


def Horizon1Model() -> ChipModel:
    """Horizon-1: rad-hard space AI on GF22FDX SOI, 18W TDP, 220 mm^2."""
    return ChipModel(
        name="Horizon-1",
        process=PROCESS_NODES["gf22fdx_soi"],
        die_area_mm2=220.0,
        package_tdp_w=18.0,
        operating_points=[
            OperatingPoint("12W_safe",   voltage_mv=700, frequency_ghz=0.80, activity_factor=0.15),
            OperatingPoint("18W_normal", voltage_mv=800, frequency_ghz=1.60, activity_factor=0.25),
            OperatingPoint("25W_burst",  voltage_mv=900, frequency_ghz=2.00, activity_factor=0.35),
        ],
        fabric=ComputeFabric(
            total_macs=8192,                            # 8 AI tiles, 32x32 systolic each w/ TMR
            supported_precisions=[
                PrecisionMode.INT8, PrecisionMode.FP16, PrecisionMode.BF16,
            ],
            peak_freq_ghz=2.00,
        ),
        memory=MemoryHierarchy(
            l1_kb=32,    l1_bw_gb_s=512.0,  l1_read_energy_pj=10.0, l1_write_energy_pj=12.0,
            l2_kb=256,   l2_bw_gb_s=180.0,  l2_read_energy_pj=28.0, l2_write_energy_pj=34.0,
            l3_kb=2048,  l3_bw_gb_s=80.0,   l3_read_energy_pj=90.0, l3_write_energy_pj=110.0,
            hbm_capacity_gb=16.0, hbm_bw_gb_s=120.0,    # LPDDR5-rad, not HBM
            hbm_read_energy_pj=350.0, hbm_write_energy_pj=420.0,
        ),
        interconnect=InterconnectFabric(
            topology="mesh", rows=3, cols=3,
            link_bits=64, link_freq_ghz=1.0,
            flit_size_bytes=8, energy_per_flit_pj=8.0,
            hop_latency_cycles=1,
        ),
    )


def Discovery1Model() -> ChipModel:
    """Discovery-1: medical-AI SoC on TSMC N3, 95W TDP, 290 mm^2."""
    return ChipModel(
        name="Discovery-1",
        process=PROCESS_NODES["tsmc_n3"],
        die_area_mm2=290.0,
        package_tdp_w=95.0,
        operating_points=[
            OperatingPoint("idle",    voltage_mv=550, frequency_ghz=1.00, activity_factor=0.10),
            OperatingPoint("low",     voltage_mv=650, frequency_ghz=2.00, activity_factor=0.20),
            OperatingPoint("nominal", voltage_mv=750, frequency_ghz=3.00, activity_factor=0.30),
            OperatingPoint("turbo",   voltage_mv=900, frequency_ghz=3.60, activity_factor=0.45),
        ],
        fabric=ComputeFabric(
            total_macs=81920,                           # 5 compute tiles x 128 MTLs x 128 PEs
            supported_precisions=[
                PrecisionMode.INT8, PrecisionMode.FP8_E4M3, PrecisionMode.FP16,
                PrecisionMode.BF16, PrecisionMode.TF32, PrecisionMode.FP32,
            ],
            peak_freq_ghz=3.60,
        ),
        memory=MemoryHierarchy(
            l1_kb=64,     l1_bw_gb_s=2048.0, l1_read_energy_pj=8.0,  l1_write_energy_pj=10.0,
            l2_kb=1024,   l2_bw_gb_s=800.0,  l2_read_energy_pj=22.0, l2_write_energy_pj=28.0,
            l3_kb=131072, l3_bw_gb_s=400.0,  l3_read_energy_pj=80.0, l3_write_energy_pj=100.0,
            hbm_capacity_gb=72.0, hbm_bw_gb_s=2400.0,    # 3x HBM3E stacks
            hbm_read_energy_pj=380.0, hbm_write_energy_pj=460.0,
        ),
        interconnect=InterconnectFabric(
            topology="mesh", rows=8, cols=8,
            link_bits=256, link_freq_ghz=2.0,
            flit_size_bytes=32, energy_per_flit_pj=14.0,
            hop_latency_cycles=1,
        ),
    )


def Nexus1Model() -> ChipModel:
    """Nexus-1 (1T): 6G RF+AI chiplet on TSMC N3, 35W TDP, 165 mm^2."""
    return ChipModel(
        name="Nexus-1",
        process=PROCESS_NODES["tsmc_n3"],
        die_area_mm2=165.0,
        package_tdp_w=35.0,
        operating_points=[
            OperatingPoint("low",     voltage_mv=600, frequency_ghz=1.20, activity_factor=0.20),
            OperatingPoint("nominal", voltage_mv=720, frequency_ghz=2.50, activity_factor=0.35),
            OperatingPoint("turbo",   voltage_mv=820, frequency_ghz=3.00, activity_factor=0.50),
        ],
        fabric=ComputeFabric(
            total_macs=4096,                            # 4 AI tiles x 32x32 systolic
            supported_precisions=[
                PrecisionMode.INT8, PrecisionMode.FP16, PrecisionMode.BF16,
            ],
            peak_freq_ghz=3.00,
        ),
        memory=MemoryHierarchy(
            l1_kb=32,    l1_bw_gb_s=960.0,  l1_read_energy_pj=8.0,  l1_write_energy_pj=10.0,
            l2_kb=512,   l2_bw_gb_s=320.0,  l2_read_energy_pj=22.0, l2_write_energy_pj=28.0,
            l3_kb=8192,  l3_bw_gb_s=160.0,  l3_read_energy_pj=80.0, l3_write_energy_pj=100.0,
            hbm_capacity_gb=8.0, hbm_bw_gb_s=120.0,     # LPDDR5
            hbm_read_energy_pj=350.0, hbm_write_energy_pj=420.0,
        ),
        interconnect=InterconnectFabric(
            topology="mesh", rows=4, cols=4,
            link_bits=128, link_freq_ghz=1.5,
            flit_size_bytes=16, energy_per_flit_pj=12.0,
            hop_latency_cycles=1,
        ),
    )


def PrometheusModel() -> ChipModel:
    """Prometheus V1: datacenter dense+sparse AI chiplet on TSMC N3,
    400W TDP per chiplet x 4 chiplets = 1600W package, 800 mm^2 per chiplet."""
    return ChipModel(
        name="Prometheus",
        process=PROCESS_NODES["tsmc_n3"],
        die_area_mm2=800.0,
        package_tdp_w=400.0,                            # Per chiplet; 4 chiplets per package
        operating_points=[
            OperatingPoint("low",      voltage_mv=600, frequency_ghz=1.50, activity_factor=0.20),
            OperatingPoint("nominal",  voltage_mv=720, frequency_ghz=2.10, activity_factor=0.40),
            OperatingPoint("perf",     voltage_mv=820, frequency_ghz=2.50, activity_factor=0.55),
            OperatingPoint("max",      voltage_mv=900, frequency_ghz=2.80, activity_factor=0.70),
        ],
        fabric=ComputeFabric(
            total_macs=786432,                          # 96 SMs x 32 warps x 256 lanes
            supported_precisions=[
                PrecisionMode.FP4_E2M1, PrecisionMode.FP8_E4M3, PrecisionMode.FP8_E5M2,
                PrecisionMode.BF16, PrecisionMode.FP16, PrecisionMode.TF32,
                PrecisionMode.FP32, PrecisionMode.FP64,
                PrecisionMode.INT4, PrecisionMode.INT8,
            ],
            peak_freq_ghz=2.80,
        ),
        memory=MemoryHierarchy(
            l1_kb=128,    l1_bw_gb_s=4096.0, l1_read_energy_pj=8.0,  l1_write_energy_pj=10.0,
            l2_kb=8192,   l2_bw_gb_s=2400.0, l2_read_energy_pj=22.0, l2_write_energy_pj=28.0,
            l3_kb=524288, l3_bw_gb_s=1200.0, l3_read_energy_pj=80.0, l3_write_energy_pj=100.0,
            hbm_capacity_gb=288.0, hbm_bw_gb_s=12300.0, # 12x HBM4 stacks at 1.025 TB/s each
            hbm_read_energy_pj=350.0, hbm_write_energy_pj=420.0,
        ),
        interconnect=InterconnectFabric(
            topology="dragonfly", rows=8, cols=8,
            link_bits=512, link_freq_ghz=2.5,
            flit_size_bytes=64, energy_per_flit_pj=18.0,
            hop_latency_cycles=2,
        ),
    )


# Convenience factory: get any chip by name
def get_chip(name: str) -> ChipModel:
    """Look up a chip model by canonical name (case-insensitive)."""
    factories = {
        "sentinel-1":  Sentinel1Model,
        "sentinel1":   Sentinel1Model,
        "horizon-1":   Horizon1Model,
        "horizon1":    Horizon1Model,
        "discovery-1": Discovery1Model,
        "discovery1":  Discovery1Model,
        "nexus-1":     Nexus1Model,
        "nexus1":      Nexus1Model,
        "prometheus":  PrometheusModel,
    }
    key = name.lower().strip()
    if key not in factories:
        raise ValueError(
            f"unknown chip '{name}'; valid: {sorted(factories.keys())}"
        )
    return factories[key]()


def all_chips():
    """Return all 5 chip models."""
    return [
        Sentinel1Model(),
        Horizon1Model(),
        Discovery1Model(),
        Nexus1Model(),
        PrometheusModel(),
    ]
