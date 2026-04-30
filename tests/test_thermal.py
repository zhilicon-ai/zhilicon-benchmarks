"""Tests for thermal model."""

from __future__ import annotations

import math
import pytest

from perf_models.thermal import (
    ThermalProfile,
    THERMAL_PROFILES,
    get_thermal_profile,
)


@pytest.fixture
def sentinel_profile():
    return THERMAL_PROFILES["Sentinel-1"]


class TestThermalProfile:
    def test_construction(self, sentinel_profile):
        assert sentinel_profile.r_jc_k_per_w == 0.18
        assert sentinel_profile.r_cs_k_per_w == 0.08
        assert sentinel_profile.r_sa_k_per_w == 0.32

    def test_r_total(self, sentinel_profile):
        assert math.isclose(sentinel_profile.r_total_k_per_w,
                            0.18 + 0.08 + 0.32, rel_tol=1e-6)

    def test_thermal_time_constant(self, sentinel_profile):
        """tau = R * C."""
        expected = sentinel_profile.r_total_k_per_w * sentinel_profile.c_th_j_per_k
        assert math.isclose(sentinel_profile.thermal_time_constant_s, expected)

    def test_default_ambient(self, sentinel_profile):
        assert sentinel_profile.t_ambient_c == 25.0

    def test_horizon_low_ambient(self):
        """Horizon-1 in space has very cold ambient."""
        h = THERMAL_PROFILES["Horizon-1"]
        assert h.t_ambient_c < 0


class TestSteadyStateTjunction:
    def test_zero_power(self, sentinel_profile):
        """At zero power, Tj should equal ambient."""
        tj = sentinel_profile.steady_state_t_junction(power_w=0.0)
        assert tj == sentinel_profile.t_ambient_c

    def test_positive_power_raises_tj(self, sentinel_profile):
        """At any positive power, Tj > T_ambient."""
        tj = sentinel_profile.steady_state_t_junction(power_w=50.0)
        assert tj > sentinel_profile.t_ambient_c

    def test_linear_in_power(self, sentinel_profile):
        """Steady-state Tj rise is linear in power."""
        tj_50 = sentinel_profile.steady_state_t_junction(power_w=50.0)
        tj_100 = sentinel_profile.steady_state_t_junction(power_w=100.0)
        rise_50 = tj_50 - sentinel_profile.t_ambient_c
        rise_100 = tj_100 - sentinel_profile.t_ambient_c
        # Should be exactly 2x
        assert math.isclose(rise_100, 2.0 * rise_50, rel_tol=1e-6)


class TestTransient:
    def test_zero_time_returns_ambient(self, sentinel_profile):
        tj = sentinel_profile.transient_t_junction(power_w=50.0, time_s=0.0)
        assert math.isclose(tj, sentinel_profile.t_ambient_c)

    def test_long_time_approaches_steady_state(self, sentinel_profile):
        """At t = 5*tau, Tj should be ~99% of steady-state rise."""
        ss = sentinel_profile.steady_state_t_junction(power_w=50.0)
        tau = sentinel_profile.thermal_time_constant_s
        tj_5tau = sentinel_profile.transient_t_junction(power_w=50.0, time_s=5*tau)
        # exp(-5) = 0.00674; so tj should be 99.3% of way to ss
        rise_to = tj_5tau - sentinel_profile.t_ambient_c
        rise_to_ss = ss - sentinel_profile.t_ambient_c
        assert rise_to / rise_to_ss > 0.99

    def test_negative_time_raises(self, sentinel_profile):
        with pytest.raises(ValueError):
            sentinel_profile.transient_t_junction(power_w=50.0, time_s=-1.0)


class TestTimeToThrottle:
    def test_low_power_no_throttle(self, sentinel_profile):
        """If steady-state Tj < throttle, returns None (never throttles)."""
        # 1 W on Sentinel-1: Tj = 25 + 0.58 = 25.58 C, far below 95
        result = sentinel_profile.time_to_throttle_s(power_w=1.0)
        assert result is None

    def test_high_power_throttles(self, sentinel_profile):
        """At a power level high enough to exceed throttle, returns positive time."""
        # 200 W: Tj_ss = 25 + 116 = 141 C, exceeds 95 throttle
        result = sentinel_profile.time_to_throttle_s(power_w=200.0)
        assert result is not None
        assert result > 0

    def test_higher_power_faster_throttle(self, sentinel_profile):
        """More power means hitting throttle sooner."""
        t1 = sentinel_profile.time_to_throttle_s(power_w=150.0)
        t2 = sentinel_profile.time_to_throttle_s(power_w=300.0)
        assert t1 is not None and t2 is not None
        assert t2 < t1


class TestIsSafeAtSteadyState:
    def test_safe_at_low_power(self, sentinel_profile):
        assert sentinel_profile.is_safe_at_steady_state(power_w=10.0)

    def test_unsafe_at_high_power(self, sentinel_profile):
        assert not sentinel_profile.is_safe_at_steady_state(power_w=500.0)

    def test_at_threshold_boundary(self, sentinel_profile):
        # T_amb + P*R = T_throttle
        # P = (T_throttle - T_amb) / R = (95-25)/0.58 ≈ 120.69 W
        threshold_p = (sentinel_profile.t_throttle_c - sentinel_profile.t_ambient_c) / sentinel_profile.r_total_k_per_w
        assert sentinel_profile.is_safe_at_steady_state(power_w=threshold_p - 0.1)
        assert not sentinel_profile.is_safe_at_steady_state(power_w=threshold_p + 0.1)


class TestThermalProfileLookup:
    def test_all_chips_have_profile(self):
        for name in ["Sentinel-1", "Horizon-1", "Discovery-1", "Nexus-1", "Prometheus"]:
            assert name in THERMAL_PROFILES
            p = get_thermal_profile(name)
            assert p.r_total_k_per_w > 0

    def test_unknown_chip_raises(self):
        with pytest.raises(KeyError):
            get_thermal_profile("UnknownChip")

    def test_prometheus_lowest_r(self):
        """Prometheus has cold-plate cooling: lowest thermal resistance."""
        p = THERMAL_PROFILES["Prometheus"]
        for other_name in ["Sentinel-1", "Horizon-1", "Discovery-1", "Nexus-1"]:
            other = THERMAL_PROFILES[other_name]
            if other_name == "Horizon-1":
                # Horizon-1 has VERY high R (vacuum, no convection)
                continue
            # Otherwise P should have lowest R
            assert p.r_total_k_per_w < other.r_total_k_per_w

    def test_horizon_highest_r(self):
        """Horizon-1 in vacuum has highest sink-to-ambient R."""
        h = THERMAL_PROFILES["Horizon-1"]
        for other_name in ["Sentinel-1", "Discovery-1", "Nexus-1", "Prometheus"]:
            other = THERMAL_PROFILES[other_name]
            assert h.r_sa_k_per_w > other.r_sa_k_per_w
