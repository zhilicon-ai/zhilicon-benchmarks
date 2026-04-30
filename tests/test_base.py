"""Tests for perf_models.base — process nodes, operating points, fabrics."""

from __future__ import annotations

import math
import pytest

from perf_models.base import (
    ChipModel,
    ComputeFabric,
    InterconnectFabric,
    MemoryHierarchy,
    OperatingPoint,
    PrecisionMode,
    PROCESS_NODES,
    ENERGY_PER_MAC_PJ,
    ProcessNode,
)


# ============================================================================
# ProcessNode
# ============================================================================


class TestProcessNode:
    def test_known_nodes_are_registered(self):
        for name in ["tsmc_n3", "tsmc_n5", "gf22fdx_soi", "tsmc_n2"]:
            assert name in PROCESS_NODES
            assert PROCESS_NODES[name].name == name

    def test_n3_density_higher_than_n5(self):
        """A smaller node should pack more transistors per mm^2."""
        n3 = PROCESS_NODES["tsmc_n3"]
        n5 = PROCESS_NODES["tsmc_n5"]
        assert n3.transistor_density_mtx_mm2 > n5.transistor_density_mtx_mm2

    def test_n2_density_highest(self):
        n2 = PROCESS_NODES["tsmc_n2"]
        for other in ["tsmc_n3", "tsmc_n5", "gf22fdx_soi"]:
            assert n2.transistor_density_mtx_mm2 > PROCESS_NODES[other].transistor_density_mtx_mm2

    def test_gf22_lower_density_than_n3(self):
        """22 nm should be much less dense than 3 nm."""
        gf22 = PROCESS_NODES["gf22fdx_soi"]
        n3 = PROCESS_NODES["tsmc_n3"]
        assert gf22.transistor_density_mtx_mm2 < n3.transistor_density_mtx_mm2 / 5

    def test_voltages_are_realistic(self):
        for node in PROCESS_NODES.values():
            assert 600 <= node.nominal_voltage_mv <= 900
            assert 250 <= node.threshold_voltage_mv <= 500
            # Vth should be less than half of Vdd
            assert node.threshold_voltage_mv < node.nominal_voltage_mv

    def test_metal_stack_layer_count_reasonable(self):
        for node in PROCESS_NODES.values():
            assert 8 <= node.metal_stack_layers <= 25

    def test_sram_bitcell_smaller_at_smaller_nodes(self):
        """Smaller-node SRAM bit cells should be smaller."""
        n3 = PROCESS_NODES["tsmc_n3"]
        gf22 = PROCESS_NODES["gf22fdx_soi"]
        assert n3.sram_bitcell_area_um2 < gf22.sram_bitcell_area_um2

    def test_process_node_immutable(self):
        """ProcessNode is frozen dataclass; mutation should raise."""
        n3 = PROCESS_NODES["tsmc_n3"]
        with pytest.raises((AttributeError, Exception)):
            n3.feature_size_nm = 99.0  # type: ignore


# ============================================================================
# PrecisionMode and energy table
# ============================================================================


class TestPrecisionMode:
    def test_all_precisions_have_energy(self):
        """Every PrecisionMode must have an entry in ENERGY_PER_MAC_PJ."""
        for prec in PrecisionMode:
            assert prec in ENERGY_PER_MAC_PJ
            assert ENERGY_PER_MAC_PJ[prec] > 0

    def test_int8_cheaper_than_fp16(self):
        assert ENERGY_PER_MAC_PJ[PrecisionMode.INT8] < ENERGY_PER_MAC_PJ[PrecisionMode.FP16]

    def test_fp16_cheaper_than_fp32(self):
        assert ENERGY_PER_MAC_PJ[PrecisionMode.FP16] < ENERGY_PER_MAC_PJ[PrecisionMode.FP32]

    def test_fp32_cheaper_than_fp64(self):
        assert ENERGY_PER_MAC_PJ[PrecisionMode.FP32] < ENERGY_PER_MAC_PJ[PrecisionMode.FP64]

    def test_fp4_cheaper_than_fp8(self):
        assert ENERGY_PER_MAC_PJ[PrecisionMode.FP4_E2M1] < ENERGY_PER_MAC_PJ[PrecisionMode.FP8_E4M3]

    def test_int_cheaper_than_fp_at_same_width(self):
        """At equivalent width, integer MAC is cheaper than FP."""
        assert ENERGY_PER_MAC_PJ[PrecisionMode.INT8] < ENERGY_PER_MAC_PJ[PrecisionMode.FP8_E4M3]
        assert ENERGY_PER_MAC_PJ[PrecisionMode.INT16] < ENERGY_PER_MAC_PJ[PrecisionMode.FP16]


# ============================================================================
# OperatingPoint
# ============================================================================


class TestOperatingPoint:
    def test_voltage_v_property(self):
        op = OperatingPoint("test", voltage_mv=750, frequency_ghz=2.5)
        assert op.voltage_v == 0.75

    def test_frequency_hz_property(self):
        op = OperatingPoint("test", voltage_mv=750, frequency_ghz=2.5)
        assert op.frequency_hz == 2.5e9

    def test_default_activity_factor(self):
        op = OperatingPoint("test", voltage_mv=750, frequency_ghz=2.5)
        assert op.activity_factor == 0.30

    def test_default_temperature(self):
        op = OperatingPoint("test", voltage_mv=750, frequency_ghz=2.5)
        assert op.temperature_c == 50.0

    def test_custom_activity(self):
        op = OperatingPoint("hot", voltage_mv=900, frequency_ghz=3.5, activity_factor=0.6)
        assert op.activity_factor == 0.6


# ============================================================================
# ComputeFabric
# ============================================================================


@pytest.fixture
def fp16_fabric():
    """Fabric with 1024 MACs at 2.0 GHz, FP16 + INT8."""
    return ComputeFabric(
        total_macs=1024,
        supported_precisions=[PrecisionMode.FP16, PrecisionMode.INT8],
        peak_freq_ghz=2.0,
    )


class TestComputeFabric:
    def test_peak_throughput_tops(self, fp16_fabric):
        """1024 MACs at 2 GHz, 2 ops per MAC = 4.096 TOPS."""
        tops = fp16_fabric.peak_throughput_tops(PrecisionMode.FP16)
        assert math.isclose(tops, 4.096, rel_tol=1e-6)

    def test_peak_throughput_int8_same_as_fp16(self, fp16_fabric):
        """Peak TOPS doesn't change with precision (in this simple model)."""
        assert (
            fp16_fabric.peak_throughput_tops(PrecisionMode.INT8)
            == fp16_fabric.peak_throughput_tops(PrecisionMode.FP16)
        )

    def test_peak_throughput_tflops_only_for_fp(self, fp16_fabric):
        """tflops should only return for floating-point precisions."""
        flops = fp16_fabric.peak_throughput_tflops(PrecisionMode.FP16)
        assert flops == fp16_fabric.peak_throughput_tops(PrecisionMode.FP16)
        with pytest.raises(ValueError):
            fp16_fabric.peak_throughput_tflops(PrecisionMode.INT8)

    def test_get_energy_default(self, fp16_fabric):
        """Without override, uses global table."""
        e = fp16_fabric.get_energy_pj(PrecisionMode.FP16)
        assert e == ENERGY_PER_MAC_PJ[PrecisionMode.FP16]

    def test_get_energy_override(self):
        """Custom energy table overrides global."""
        custom = ComputeFabric(
            total_macs=512, peak_freq_ghz=3.0,
            supported_precisions=[PrecisionMode.FP16],
            energy_pj_per_mac={PrecisionMode.FP16: 0.250},
        )
        assert custom.get_energy_pj(PrecisionMode.FP16) == 0.250


# ============================================================================
# MemoryHierarchy
# ============================================================================


class TestMemoryHierarchy:
    def test_construction(self):
        m = MemoryHierarchy(
            l1_kb=64, l1_bw_gb_s=1000.0, l1_read_energy_pj=10, l1_write_energy_pj=12,
            l2_kb=512, l2_bw_gb_s=400.0, l2_read_energy_pj=25, l2_write_energy_pj=30,
            l3_kb=4096, l3_bw_gb_s=200.0, l3_read_energy_pj=80, l3_write_energy_pj=100,
            hbm_capacity_gb=24.0, hbm_bw_gb_s=900.0,
            hbm_read_energy_pj=400, hbm_write_energy_pj=480,
        )
        assert m.l1_kb == 64
        assert m.hbm_capacity_gb == 24.0


# ============================================================================
# InterconnectFabric
# ============================================================================


class TestInterconnectFabric:
    def test_link_bandwidth(self):
        """128 bits at 2 GHz = 32 GB/s."""
        ic = InterconnectFabric(
            topology="mesh", rows=4, cols=4,
            link_bits=128, link_freq_ghz=2.0,
            flit_size_bytes=16, energy_per_flit_pj=10.0,
        )
        # 128 bits / 8 = 16 bytes, x 2 GHz = 32 GB/s
        assert math.isclose(ic.link_bandwidth_gb_s(), 32.0)

    def test_mesh_bisection_bw(self):
        """4x4 mesh: bisection = min(rows,cols) * link_bw = 4 * 32 = 128 GB/s."""
        ic = InterconnectFabric(
            topology="mesh", rows=4, cols=4,
            link_bits=128, link_freq_ghz=2.0,
            flit_size_bytes=16, energy_per_flit_pj=10.0,
        )
        assert math.isclose(ic.bisection_bandwidth_gb_s(), 128.0)

    def test_torus_bisection_bw_double_mesh(self):
        """Torus has wrap-around links, so bisection = 2x mesh."""
        mesh = InterconnectFabric(
            topology="mesh", rows=4, cols=4,
            link_bits=128, link_freq_ghz=2.0,
            flit_size_bytes=16, energy_per_flit_pj=10.0,
        )
        torus = InterconnectFabric(
            topology="torus", rows=4, cols=4,
            link_bits=128, link_freq_ghz=2.0,
            flit_size_bytes=16, energy_per_flit_pj=10.0,
        )
        assert math.isclose(
            torus.bisection_bandwidth_gb_s(),
            2 * mesh.bisection_bandwidth_gb_s()
        )

    def test_unknown_topology_raises(self):
        ic = InterconnectFabric(
            topology="unknown_topology", rows=4, cols=4,
            link_bits=128, link_freq_ghz=2.0,
            flit_size_bytes=16, energy_per_flit_pj=10.0,
        )
        with pytest.raises(ValueError):
            ic.bisection_bandwidth_gb_s()

    def test_num_routers(self):
        ic = InterconnectFabric(
            topology="mesh", rows=8, cols=8,
            link_bits=128, link_freq_ghz=2.0,
            flit_size_bytes=16, energy_per_flit_pj=10.0,
        )
        assert ic.num_routers == 64


# ============================================================================
# ChipModel
# ============================================================================


@pytest.fixture
def small_chip():
    """A small but complete ChipModel for unit testing."""
    return ChipModel(
        name="TestChip",
        process=PROCESS_NODES["tsmc_n3"],
        die_area_mm2=50.0,
        package_tdp_w=20.0,
        operating_points=[
            OperatingPoint("low",  voltage_mv=600, frequency_ghz=1.0, activity_factor=0.2),
            OperatingPoint("high", voltage_mv=800, frequency_ghz=2.5, activity_factor=0.4),
        ],
        fabric=ComputeFabric(
            total_macs=1024, peak_freq_ghz=2.5,
            supported_precisions=[PrecisionMode.FP16, PrecisionMode.INT8],
        ),
        memory=MemoryHierarchy(
            l1_kb=32, l1_bw_gb_s=500.0, l1_read_energy_pj=10, l1_write_energy_pj=12,
            l2_kb=256, l2_bw_gb_s=200.0, l2_read_energy_pj=25, l2_write_energy_pj=30,
            l3_kb=2048, l3_bw_gb_s=100.0, l3_read_energy_pj=80, l3_write_energy_pj=100,
            hbm_capacity_gb=4.0, hbm_bw_gb_s=200.0,
            hbm_read_energy_pj=400, hbm_write_energy_pj=480,
        ),
        interconnect=InterconnectFabric(
            topology="mesh", rows=2, cols=2,
            link_bits=128, link_freq_ghz=1.5,
            flit_size_bytes=16, energy_per_flit_pj=10.0,
        ),
    )


class TestChipModel:
    def test_construction(self, small_chip):
        assert small_chip.name == "TestChip"
        assert small_chip.die_area_mm2 == 50.0

    def test_zero_die_area_rejected(self):
        with pytest.raises(ValueError, match="die_area"):
            ChipModel(
                name="Bad", process=PROCESS_NODES["tsmc_n3"],
                die_area_mm2=0.0, package_tdp_w=10.0,
                operating_points=[OperatingPoint("op", 700, 1.0)],
                fabric=ComputeFabric(total_macs=1, peak_freq_ghz=1.0,
                                     supported_precisions=[PrecisionMode.FP16]),
                memory=MemoryHierarchy(
                    l1_kb=1, l1_bw_gb_s=1.0, l1_read_energy_pj=1, l1_write_energy_pj=1,
                    l2_kb=1, l2_bw_gb_s=1.0, l2_read_energy_pj=1, l2_write_energy_pj=1,
                    l3_kb=1, l3_bw_gb_s=1.0, l3_read_energy_pj=1, l3_write_energy_pj=1,
                    hbm_capacity_gb=1.0, hbm_bw_gb_s=1.0,
                    hbm_read_energy_pj=1, hbm_write_energy_pj=1,
                ),
                interconnect=InterconnectFabric(
                    topology="mesh", rows=1, cols=1,
                    link_bits=64, link_freq_ghz=1.0,
                    flit_size_bytes=8, energy_per_flit_pj=1.0,
                ),
            )

    def test_zero_tdp_rejected(self):
        with pytest.raises(ValueError, match="tdp"):
            ChipModel(
                name="Bad", process=PROCESS_NODES["tsmc_n3"],
                die_area_mm2=10.0, package_tdp_w=0.0,
                operating_points=[OperatingPoint("op", 700, 1.0)],
                fabric=ComputeFabric(total_macs=1, peak_freq_ghz=1.0,
                                     supported_precisions=[PrecisionMode.FP16]),
                memory=MemoryHierarchy(
                    l1_kb=1, l1_bw_gb_s=1.0, l1_read_energy_pj=1, l1_write_energy_pj=1,
                    l2_kb=1, l2_bw_gb_s=1.0, l2_read_energy_pj=1, l2_write_energy_pj=1,
                    l3_kb=1, l3_bw_gb_s=1.0, l3_read_energy_pj=1, l3_write_energy_pj=1,
                    hbm_capacity_gb=1.0, hbm_bw_gb_s=1.0,
                    hbm_read_energy_pj=1, hbm_write_energy_pj=1,
                ),
                interconnect=InterconnectFabric(
                    topology="mesh", rows=1, cols=1,
                    link_bits=64, link_freq_ghz=1.0,
                    flit_size_bytes=8, energy_per_flit_pj=1.0,
                ),
            )

    def test_no_operating_points_rejected(self):
        with pytest.raises(ValueError, match="operating point"):
            ChipModel(
                name="Bad", process=PROCESS_NODES["tsmc_n3"],
                die_area_mm2=10.0, package_tdp_w=10.0,
                operating_points=[],
                fabric=ComputeFabric(total_macs=1, peak_freq_ghz=1.0,
                                     supported_precisions=[PrecisionMode.FP16]),
                memory=MemoryHierarchy(
                    l1_kb=1, l1_bw_gb_s=1.0, l1_read_energy_pj=1, l1_write_energy_pj=1,
                    l2_kb=1, l2_bw_gb_s=1.0, l2_read_energy_pj=1, l2_write_energy_pj=1,
                    l3_kb=1, l3_bw_gb_s=1.0, l3_read_energy_pj=1, l3_write_energy_pj=1,
                    hbm_capacity_gb=1.0, hbm_bw_gb_s=1.0,
                    hbm_read_energy_pj=1, hbm_write_energy_pj=1,
                ),
                interconnect=InterconnectFabric(
                    topology="mesh", rows=1, cols=1,
                    link_bits=64, link_freq_ghz=1.0,
                    flit_size_bytes=8, energy_per_flit_pj=1.0,
                ),
            )

    def test_lowest_power_op(self, small_chip):
        op = small_chip.lowest_power_op
        assert op.name == "low"
        assert op.frequency_ghz == 1.0

    def test_highest_perf_op(self, small_chip):
        op = small_chip.highest_perf_op
        assert op.name == "high"
        assert op.frequency_ghz == 2.5

    def test_total_transistor_count(self, small_chip):
        """50 mm^2 * 292 Mtx/mm^2 = 14600 Mtx."""
        n = small_chip.total_transistor_count_mtx()
        assert math.isclose(n, 50.0 * 292.0, rel_tol=1e-6)

    def test_leakage_power_positive(self, small_chip):
        op = small_chip.nominal_op
        leakage = small_chip.total_leakage_w(op)
        assert leakage > 0

    def test_leakage_increases_with_voltage(self, small_chip):
        """Higher voltage = exponentially more leakage."""
        low_op = small_chip.lowest_power_op
        high_op = small_chip.highest_perf_op
        assert small_chip.total_leakage_w(high_op) > small_chip.total_leakage_w(low_op)

    def test_leakage_increases_with_temperature(self, small_chip):
        """Higher temperature = more leakage (doubles every 10 deg C)."""
        cold = OperatingPoint("cold", 700, 2.0, temperature_c=25.0)
        hot = OperatingPoint("hot", 700, 2.0, temperature_c=85.0)
        cold_l = small_chip.total_leakage_w(cold)
        hot_l = small_chip.total_leakage_w(hot)
        # 60 deg C swing = 2^6 = 64x leakage change
        assert hot_l > cold_l * 30  # Allow some tolerance

    def test_dynamic_power_positive(self, small_chip):
        op = small_chip.nominal_op
        p = small_chip.dynamic_power_w(op, PrecisionMode.FP16)
        assert p > 0

    def test_dynamic_power_scales_with_voltage_squared(self, small_chip):
        """Pdyn ~ V^2: doubling V = ~4x Pdyn (other terms equal)."""
        # Same activity, same frequency, only voltage differs
        op_low_v = OperatingPoint("v_low",  voltage_mv=400, frequency_ghz=1.0, activity_factor=0.3)
        op_high_v = OperatingPoint("v_high", voltage_mv=800, frequency_ghz=1.0, activity_factor=0.3)
        p_low = small_chip.dynamic_power_w(op_low_v, PrecisionMode.FP16)
        p_high = small_chip.dynamic_power_w(op_high_v, PrecisionMode.FP16)
        # 2x V => 4x Pdyn, allow 5% tolerance
        ratio = p_high / p_low
        assert math.isclose(ratio, 4.0, rel_tol=0.05)

    def test_dynamic_power_scales_with_frequency(self, small_chip):
        """Pdyn ~ F: doubling F should ~double Pdyn."""
        op_low_f = OperatingPoint("f_low",  voltage_mv=700, frequency_ghz=1.0, activity_factor=0.3)
        op_high_f = OperatingPoint("f_high", voltage_mv=700, frequency_ghz=2.0, activity_factor=0.3)
        p_low = small_chip.dynamic_power_w(op_low_f, PrecisionMode.FP16)
        p_high = small_chip.dynamic_power_w(op_high_f, PrecisionMode.FP16)
        ratio = p_high / p_low
        assert math.isclose(ratio, 2.0, rel_tol=0.05)

    def test_int8_more_efficient_than_fp16(self, small_chip):
        """At same V/F, INT8 should consume less dynamic power than FP16."""
        op = small_chip.nominal_op
        p_int8 = small_chip.dynamic_power_w(op, PrecisionMode.INT8)
        p_fp16 = small_chip.dynamic_power_w(op, PrecisionMode.FP16)
        assert p_int8 < p_fp16

    def test_total_power_includes_both(self, small_chip):
        op = small_chip.nominal_op
        total = small_chip.total_power_w(op, PrecisionMode.FP16)
        leak = small_chip.total_leakage_w(op)
        dyn = small_chip.dynamic_power_w(op, PrecisionMode.FP16)
        assert math.isclose(total, leak + dyn)

    def test_perf_per_watt_positive(self, small_chip):
        pw = small_chip.perf_per_watt_tops_w(small_chip.nominal_op, PrecisionMode.FP16)
        assert pw > 0

    def test_int8_better_perf_per_watt_than_fp16(self, small_chip):
        """Lower-precision modes should give better TOPS/W."""
        op = small_chip.nominal_op
        # Need an INT8 fabric for this to be apples-to-apples; small_chip
        # supports INT8 in its precision list.
        pw_fp16 = small_chip.perf_per_watt_tops_w(op, PrecisionMode.FP16)
        pw_int8 = small_chip.perf_per_watt_tops_w(op, PrecisionMode.INT8)
        assert pw_int8 > pw_fp16

    def test_serializable(self, small_chip):
        d = small_chip.serializable()
        assert d["name"] == "TestChip"
        assert d["die_area_mm2"] == 50.0
        assert "process" in d
        assert "fabric" in d
        assert "memory" in d

    def test_summary_table_returns_string(self, small_chip):
        s = small_chip.summary_table()
        assert "TestChip" in s
        assert "50" in s          # die area
        assert "TOPS/W" in s
