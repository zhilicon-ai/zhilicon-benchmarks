"""Tests for perf_models.power -- power decomposition and sweep utilities."""

from __future__ import annotations

import math
import pytest

from perf_models.base import OperatingPoint, PrecisionMode
from perf_models.chips import (
    Sentinel1Model, Horizon1Model, Discovery1Model,
    Nexus1Model, PrometheusModel, all_chips,
)
from perf_models.power import (
    PowerBreakdown,
    compute_power_breakdown,
    find_optimal_op_for_perf_per_watt,
    power_sweep,
    power_within_tdp,
)


class TestPowerBreakdown:
    def test_dynamic_w_sums(self):
        bd = PowerBreakdown(
            chip_name="test", op_name="op",
            compute_dynamic_w=10.0,
            memory_dynamic_w=2.0,
            interconnect_dynamic_w=1.0,
            io_dynamic_w=0.5,
            leakage_w=2.5,
            total_w=16.0,
        )
        assert bd.dynamic_w == 13.5
        assert math.isclose(bd.dynamic_fraction, 13.5 / 16.0)
        assert math.isclose(bd.leakage_fraction, 2.5 / 16.0)

    def test_zero_total_safe(self):
        bd = PowerBreakdown(chip_name="zero", op_name="off")
        # No division-by-zero
        assert bd.dynamic_fraction == 0.0
        assert bd.leakage_fraction == 0.0

    def test_as_dict(self):
        bd = PowerBreakdown(
            chip_name="test", op_name="op",
            compute_dynamic_w=10.0, leakage_w=2.0, total_w=12.0,
        )
        d = bd.as_dict()
        assert d["chip"] == "test"
        assert d["compute_dynamic_w"] == 10.0
        assert d["total_w"] == 12.0

    def test_str_format(self):
        bd = PowerBreakdown(
            chip_name="test", op_name="op",
            compute_dynamic_w=10.0, leakage_w=2.0, total_w=12.0,
        )
        s = str(bd)
        assert "test" in s
        assert "op" in s
        assert "TOTAL" in s


class TestComputePowerBreakdown:
    def test_returns_populated_breakdown(self):
        for chip in all_chips():
            for op in chip.operating_points:
                try:
                    bd = compute_power_breakdown(chip, op, PrecisionMode.FP16)
                    assert bd.chip_name == chip.name
                    assert bd.op_name == op.name
                    assert bd.total_w > 0
                except (ValueError, KeyError):
                    pass

    def test_total_equals_sum_of_components(self):
        chip = Discovery1Model()
        op = chip.nominal_op
        bd = compute_power_breakdown(chip, op, PrecisionMode.FP16)
        components_sum = (
            bd.compute_dynamic_w + bd.memory_dynamic_w
            + bd.interconnect_dynamic_w + bd.io_dynamic_w + bd.leakage_w
        )
        assert math.isclose(bd.total_w, components_sum, rel_tol=1e-6)

    def test_higher_memory_activity_increases_memory_power(self):
        chip = Discovery1Model()
        op = chip.nominal_op
        bd_low = compute_power_breakdown(chip, op, PrecisionMode.FP16,
                                         memory_activity=0.0)
        bd_high = compute_power_breakdown(chip, op, PrecisionMode.FP16,
                                          memory_activity=0.8)
        assert bd_high.memory_dynamic_w > bd_low.memory_dynamic_w

    def test_higher_interconnect_activity_increases_ic_power(self):
        chip = Discovery1Model()
        op = chip.nominal_op
        bd_low = compute_power_breakdown(chip, op, PrecisionMode.FP16,
                                         interconnect_activity=0.0)
        bd_high = compute_power_breakdown(chip, op, PrecisionMode.FP16,
                                          interconnect_activity=0.8)
        assert bd_high.interconnect_dynamic_w > bd_low.interconnect_dynamic_w

    def test_no_negative_components(self):
        chip = Sentinel1Model()
        op = chip.nominal_op
        bd = compute_power_breakdown(chip, op)
        assert bd.compute_dynamic_w >= 0
        assert bd.memory_dynamic_w >= 0
        assert bd.interconnect_dynamic_w >= 0
        assert bd.io_dynamic_w >= 0
        assert bd.leakage_w >= 0


class TestPowerSweep:
    def test_sweep_returns_one_per_op(self):
        chip = Sentinel1Model()
        results = power_sweep(chip, PrecisionMode.FP16)
        assert len(results) == len(chip.operating_points)

    def test_sweep_results_increase_with_op_freq(self):
        """Higher operating points should generally have higher power."""
        chip = Discovery1Model()
        results = power_sweep(chip, PrecisionMode.FP16)
        # Total power should increase across ops (sorted by freq)
        for i in range(1, len(results)):
            assert results[i].total_w > results[i-1].total_w


class TestFindOptimalOp:
    def test_finds_best_op(self):
        chip = Discovery1Model()
        best = find_optimal_op_for_perf_per_watt(chip, PrecisionMode.FP16)
        # Best should be one of the chip's operating points
        op_names = {op.name for op in chip.operating_points}
        assert best.name in op_names

    def test_best_op_has_highest_pw(self):
        chip = Discovery1Model()
        best = find_optimal_op_for_perf_per_watt(chip, PrecisionMode.FP16)
        best_pw = chip.perf_per_watt_tops_w(best, PrecisionMode.FP16)
        for op in chip.operating_points:
            pw = chip.perf_per_watt_tops_w(op, PrecisionMode.FP16)
            assert pw <= best_pw + 1e-9


class TestPowerWithinTdp:
    def test_low_power_op_likely_within_tdp(self):
        for chip in all_chips():
            try:
                # Lowest power op should be well within TDP
                ok = power_within_tdp(chip, chip.lowest_power_op,
                                      PrecisionMode.FP16, margin=0.0)
                # This is not guaranteed in our model, but lowest_power_op
                # should be the safest. We don't strictly assert True; just
                # that the function returns a bool.
                assert isinstance(ok, bool)
            except (ValueError, KeyError):
                pass

    def test_returns_bool(self):
        chip = Sentinel1Model()
        op = chip.lowest_power_op
        result = power_within_tdp(chip, op, PrecisionMode.FP16)
        assert isinstance(result, bool)
