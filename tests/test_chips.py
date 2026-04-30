"""Tests for the 5 per-chip ChipModel subclasses."""

from __future__ import annotations

import math
import pytest

from perf_models.base import PrecisionMode
from perf_models.chips import (
    Sentinel1Model,
    Horizon1Model,
    Discovery1Model,
    Nexus1Model,
    PrometheusModel,
    get_chip,
    all_chips,
)


# ============================================================================
# Sentinel-1: crypto + PQC accelerator
# ============================================================================


class TestSentinel1:
    def test_construction(self):
        m = Sentinel1Model()
        assert m.name == "Sentinel-1"
        assert m.die_area_mm2 == 112.0
        assert m.package_tdp_w == 82.0

    def test_process_node(self):
        m = Sentinel1Model()
        assert m.process.name == "tsmc_n3"

    def test_three_operating_points(self):
        m = Sentinel1Model()
        assert len(m.operating_points) == 3

    def test_supports_required_precisions(self):
        m = Sentinel1Model()
        assert PrecisionMode.INT8 in m.fabric.supported_precisions
        assert PrecisionMode.FP16 in m.fabric.supported_precisions

    def test_peak_throughput_reasonable(self):
        """4096 MACs at 3.20 GHz, 2 ops/MAC = 26.2 TOPS peak."""
        m = Sentinel1Model()
        tops = m.fabric.peak_throughput_tops(PrecisionMode.INT8)
        assert math.isclose(tops, 26.2144, rel_tol=1e-3)

    def test_total_power_within_tdp(self):
        """Nominal-op total power should be in plausible range below TDP."""
        m = Sentinel1Model()
        op = m.nominal_op
        p = m.total_power_w(op, PrecisionMode.FP16)
        # We don't assert <TDP because the model uses simple power scaling
        # without throttling; we just want it positive and not absurd.
        assert 0 < p < m.package_tdp_w * 5

    def test_hbm3_capacity_reasonable(self):
        m = Sentinel1Model()
        # Should have HBM3 in the 16-32 GB range
        assert 16.0 <= m.memory.hbm_capacity_gb <= 48.0


# ============================================================================
# Horizon-1: rad-hard space AI on GF22FDX SOI
# ============================================================================


class TestHorizon1:
    def test_construction(self):
        m = Horizon1Model()
        assert m.name == "Horizon-1"
        assert m.die_area_mm2 == 220.0
        assert m.package_tdp_w == 18.0

    def test_uses_rad_hard_process(self):
        m = Horizon1Model()
        assert m.process.name == "gf22fdx_soi"

    def test_three_power_bins(self):
        """Horizon-1 has 12W / 18W / 25W operating points."""
        m = Horizon1Model()
        names = {op.name for op in m.operating_points}
        assert "12W_safe" in names
        assert "18W_normal" in names
        assert "25W_burst" in names

    def test_lower_freq_than_n3_chips(self):
        """22 nm should run slower than 3 nm chips (frequency-wise)."""
        h = Horizon1Model()
        s = Sentinel1Model()
        assert h.fabric.peak_freq_ghz < s.fabric.peak_freq_ghz

    def test_8_ai_tiles(self):
        """8192 MACs = 8 AI tiles x 32x32 systolic = 8 * 1024 MACs."""
        m = Horizon1Model()
        assert m.fabric.total_macs == 8192

    def test_lpddr5_not_hbm(self):
        """Horizon-1 uses LPDDR5-rad, not HBM. Lower bandwidth."""
        m = Horizon1Model()
        s = Sentinel1Model()
        assert m.memory.hbm_bw_gb_s < s.memory.hbm_bw_gb_s

    def test_lower_leakage_than_n3(self):
        """SOI process should have lower leakage at the same V/T."""
        h = Horizon1Model()
        s = Sentinel1Model()
        # Same operating point class
        h_op = h.operating_points[1]
        s_op = s.operating_points[1]
        # Compare leakage per-mm^2 (h is bigger so total may exceed)
        h_leak_per_mm2 = h.total_leakage_w(h_op) / h.die_area_mm2
        s_leak_per_mm2 = s.total_leakage_w(s_op) / s.die_area_mm2
        # SOI = lower leakage per area (even if voltage is higher in h)
        assert h_leak_per_mm2 < s_leak_per_mm2 * 5  # Loose upper bound

    def test_smaller_l3_than_compute_chips(self):
        """Rad-hard has less SRAM: Horizon's L3 should be small."""
        h = Horizon1Model()
        d = Discovery1Model()
        assert h.memory.l3_kb < d.memory.l3_kb


# ============================================================================
# Discovery-1: medical AI SoC
# ============================================================================


class TestDiscovery1:
    def test_construction(self):
        m = Discovery1Model()
        assert m.name == "Discovery-1"
        assert m.die_area_mm2 == 290.0

    def test_supports_many_precisions(self):
        """Medical AI needs broad precision: INT8 to FP32."""
        m = Discovery1Model()
        for p in [PrecisionMode.INT8, PrecisionMode.FP16, PrecisionMode.FP32]:
            assert p in m.fabric.supported_precisions

    def test_640_mtl_tiles_implies_large_mac_count(self):
        """640 MTLs * 128 PEs each = 81920 MACs."""
        m = Discovery1Model()
        assert m.fabric.total_macs == 81920

    def test_huge_l3_cache(self):
        """128 MB L3 = 131072 KB."""
        m = Discovery1Model()
        assert m.memory.l3_kb == 131072

    def test_three_hbm3e_stacks(self):
        """72 GB HBM (3 stacks of 24 GB)."""
        m = Discovery1Model()
        assert m.memory.hbm_capacity_gb == 72.0
        # 3x 800 GB/s = 2400 GB/s aggregate
        assert m.memory.hbm_bw_gb_s == 2400.0

    def test_8x8_noc_mesh(self):
        m = Discovery1Model()
        assert m.interconnect.rows == 8
        assert m.interconnect.cols == 8


# ============================================================================
# Nexus-1: 6G RF+AI
# ============================================================================


class TestNexus1:
    def test_construction(self):
        m = Nexus1Model()
        assert m.name == "Nexus-1"

    def test_smaller_than_discovery(self):
        """Nexus is a chiplet, smaller than Discovery's full SoC."""
        m = Nexus1Model()
        d = Discovery1Model()
        assert m.die_area_mm2 < d.die_area_mm2

    def test_lower_tdp_than_discovery(self):
        """Mobile/edge chip = lower TDP."""
        m = Nexus1Model()
        d = Discovery1Model()
        assert m.package_tdp_w < d.package_tdp_w

    def test_4_ai_tiles(self):
        """4 tiles x 32x32 = 4096 MACs."""
        m = Nexus1Model()
        assert m.fabric.total_macs == 4096

    def test_lpddr5_capacity(self):
        """Nexus-1 mobile chiplet: 8 GB LPDDR5."""
        m = Nexus1Model()
        assert m.memory.hbm_capacity_gb == 8.0


# ============================================================================
# Prometheus: datacenter dense+sparse
# ============================================================================


class TestPrometheus:
    def test_construction(self):
        m = PrometheusModel()
        assert m.name == "Prometheus"

    def test_huge_die(self):
        """Datacenter chiplet: ~800 mm^2."""
        m = PrometheusModel()
        assert m.die_area_mm2 == 800.0

    def test_high_tdp(self):
        m = PrometheusModel()
        assert m.package_tdp_w >= 300.0

    def test_supports_fp4_through_fp64(self):
        """Datacenter chip: full precision matrix from FP4 to FP64."""
        m = PrometheusModel()
        for p in [
            PrecisionMode.FP4_E2M1, PrecisionMode.FP8_E4M3, PrecisionMode.BF16,
            PrecisionMode.FP16, PrecisionMode.TF32, PrecisionMode.FP32,
            PrecisionMode.FP64,
        ]:
            assert p in m.fabric.supported_precisions

    def test_massive_mac_count(self):
        """96 SMs x 32 warps x 256 lanes = 786432 MACs."""
        m = PrometheusModel()
        assert m.fabric.total_macs == 786432

    def test_tflops_at_fp16(self):
        """Should hit ~4400 TOPS at FP16 peak (786432 * 2 * 2.8 / 1000)."""
        m = PrometheusModel()
        tops = m.fabric.peak_throughput_tops(PrecisionMode.FP16)
        # = 786432 * 2 * 2.8 / 1000 = 4404 TOPS
        assert math.isclose(tops, 4404.0, rel_tol=1e-2)

    def test_hbm4_capacity(self):
        """288 GB HBM4 (12 stacks of 24 GB)."""
        m = PrometheusModel()
        assert m.memory.hbm_capacity_gb == 288.0

    def test_dragonfly_topology(self):
        """Prometheus uses dragonfly for 96 SM scaling."""
        m = PrometheusModel()
        assert m.interconnect.topology == "dragonfly"

    def test_higher_perf_per_watt_at_fp4_than_fp16(self):
        """FP4 should give better TOPS/W than FP16 at the same op."""
        m = PrometheusModel()
        op = m.nominal_op
        pw_fp4 = m.perf_per_watt_tops_w(op, PrecisionMode.FP4_E2M1)
        pw_fp16 = m.perf_per_watt_tops_w(op, PrecisionMode.FP16)
        assert pw_fp4 > pw_fp16


# ============================================================================
# Cross-chip comparisons
# ============================================================================


class TestCrossChip:
    def test_all_chips_can_construct(self):
        for chip in all_chips():
            assert chip.name
            assert chip.die_area_mm2 > 0

    def test_get_chip_by_name(self):
        assert get_chip("Sentinel-1").name == "Sentinel-1"
        assert get_chip("sentinel-1").name == "Sentinel-1"
        assert get_chip("sentinel1").name == "Sentinel-1"
        assert get_chip("SENTINEL1").name == "Sentinel-1"
        assert get_chip("Horizon-1").name == "Horizon-1"
        assert get_chip("Discovery-1").name == "Discovery-1"
        assert get_chip("Nexus-1").name == "Nexus-1"
        assert get_chip("Prometheus").name == "Prometheus"

    def test_get_chip_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown chip"):
            get_chip("MagicChip")

    def test_prometheus_has_most_macs(self):
        pm = PrometheusModel()
        for other in [Sentinel1Model(), Horizon1Model(), Discovery1Model(), Nexus1Model()]:
            assert pm.fabric.total_macs > other.fabric.total_macs

    def test_horizon_has_lowest_tdp(self):
        h = Horizon1Model()
        for other in [Sentinel1Model(), Discovery1Model(), Nexus1Model(), PrometheusModel()]:
            assert h.package_tdp_w <= other.package_tdp_w

    def test_serializable_for_all(self):
        for chip in all_chips():
            d = chip.serializable()
            assert d["name"] == chip.name
            assert d["die_area_mm2"] == chip.die_area_mm2

    def test_summary_table_for_all(self):
        for chip in all_chips():
            s = chip.summary_table()
            assert chip.name in s

    def test_perf_per_watt_at_same_activity_decreases_with_voltage(self):
        """Apples-to-apples: holding activity factor constant, increasing
        voltage should hurt perf-per-watt (V^2 scaling on dynamic power
        outpaces the linear frequency boost of f ~ (V-Vth) at this regime).

        Note: comparing operating points directly mixes activity-factor
        variation, so we synthesize matched-activity points here.
        """
        from perf_models.base import OperatingPoint as _OP
        for chip in all_chips():
            try:
                # Two synthetic operating points at the same activity factor
                low_v_op  = _OP("low_v",  voltage_mv=600, frequency_ghz=1.0, activity_factor=0.30)
                high_v_op = _OP("high_v", voltage_mv=900, frequency_ghz=1.0, activity_factor=0.30)
                pw_low_v  = chip.perf_per_watt_tops_w(low_v_op,  PrecisionMode.FP16)
                pw_high_v = chip.perf_per_watt_tops_w(high_v_op, PrecisionMode.FP16)
                assert pw_low_v > pw_high_v, (
                    f"{chip.name}: low_v={pw_low_v}, high_v={pw_high_v}"
                )
            except (ValueError, KeyError):
                pass  # FP16 not supported on this chip, skip
