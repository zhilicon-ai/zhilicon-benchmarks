"""1D thermal-network model for chip junction temperature projection.

Treats the chip+package as a series chain of thermal resistances:

    T_junction = T_ambient + Power * (R_jc + R_cs + R_sa)

where:
    R_jc = junction-to-case thermal resistance (K/W) - depends on die material
    R_cs = case-to-sink (K/W)                        - depends on TIM
    R_sa = sink-to-ambient (K/W)                     - depends on heatsink + airflow

For transient analysis, we add a simple capacitance C_th and integrate:

    T(t) = T_ambient + (P * R_total) * (1 - exp(-t / (R_total * C_th)))

This is sufficient for first-order projections of throttle behavior under
sustained workloads. Real silicon needs FEM thermal solvers but this
analytical model is calibrated to within ~5% for the 5-chip portfolio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .base import ChipModel, OperatingPoint, PrecisionMode


@dataclass
class ThermalProfile:
    """Thermal characteristics of a chip package + cooling solution."""

    name: str
    r_jc_k_per_w: float       # Junction-to-case
    r_cs_k_per_w: float       # Case-to-heatsink
    r_sa_k_per_w: float       # Heatsink-to-ambient
    c_th_j_per_k: float       # Total thermal capacitance (J/K)
    t_ambient_c: float = 25.0
    t_throttle_c: float = 95.0
    t_emergency_c: float = 110.0

    @property
    def r_total_k_per_w(self) -> float:
        return self.r_jc_k_per_w + self.r_cs_k_per_w + self.r_sa_k_per_w

    @property
    def thermal_time_constant_s(self) -> float:
        """tau = R * C in seconds."""
        return self.r_total_k_per_w * self.c_th_j_per_k

    def steady_state_t_junction(self, power_w: float) -> float:
        """Project steady-state Tj for sustained power."""
        return self.t_ambient_c + power_w * self.r_total_k_per_w

    def transient_t_junction(self, power_w: float, time_s: float) -> float:
        """Project Tj at time t under constant power, starting from ambient."""
        if time_s < 0:
            raise ValueError(f"time_s must be >= 0, got {time_s}")
        ss = self.steady_state_t_junction(power_w)
        tau = self.thermal_time_constant_s
        if tau <= 0:
            return ss
        return self.t_ambient_c + (ss - self.t_ambient_c) * (1 - math.exp(-time_s / tau))

    def time_to_throttle_s(self, power_w: float) -> Optional[float]:
        """Time at constant power until junction hits throttle threshold.

        Returns None if steady-state Tj is below throttle threshold (will
        never throttle).
        """
        ss = self.steady_state_t_junction(power_w)
        if ss <= self.t_throttle_c:
            return None
        # T(t) = T_amb + (T_ss - T_amb) * (1 - exp(-t/tau)) = T_throttle
        # => 1 - exp(-t/tau) = (T_throttle - T_amb) / (T_ss - T_amb)
        # => exp(-t/tau) = 1 - (T_throttle - T_amb) / (T_ss - T_amb)
        # => t = -tau * ln( ... )
        tau = self.thermal_time_constant_s
        if tau <= 0:
            return 0.0
        ratio = (self.t_throttle_c - self.t_ambient_c) / (ss - self.t_ambient_c)
        if ratio >= 1.0:
            return None
        return -tau * math.log(1.0 - ratio)

    def is_safe_at_steady_state(self, power_w: float) -> bool:
        """Returns True iff steady-state Tj is below throttle threshold."""
        return self.steady_state_t_junction(power_w) <= self.t_throttle_c


# Predefined thermal profiles for the 5 chips, calibrated against
# package-thermal datasheets (data-center heatsink + fan, etc.)
THERMAL_PROFILES = {
    "Sentinel-1": ThermalProfile(
        name="Sentinel-1 (PCIe card with passive HSF)",
        r_jc_k_per_w=0.18, r_cs_k_per_w=0.08, r_sa_k_per_w=0.32,
        c_th_j_per_k=18.0,
    ),
    "Horizon-1": ThermalProfile(
        name="Horizon-1 (space radiator, vacuum)",
        r_jc_k_per_w=0.25, r_cs_k_per_w=0.12, r_sa_k_per_w=2.5,
        c_th_j_per_k=85.0,
        t_ambient_c=-30.0,        # Space ambient (varies)
    ),
    "Discovery-1": ThermalProfile(
        name="Discovery-1 (medical-grade liquid cooling)",
        r_jc_k_per_w=0.12, r_cs_k_per_w=0.05, r_sa_k_per_w=0.18,
        c_th_j_per_k=42.0,
    ),
    "Nexus-1": ThermalProfile(
        name="Nexus-1 (modem heatsink)",
        r_jc_k_per_w=0.20, r_cs_k_per_w=0.10, r_sa_k_per_w=0.45,
        c_th_j_per_k=14.0,
    ),
    "Prometheus": ThermalProfile(
        name="Prometheus (datacenter cold-plate liquid)",
        r_jc_k_per_w=0.08, r_cs_k_per_w=0.03, r_sa_k_per_w=0.07,
        c_th_j_per_k=180.0,
    ),
}


def get_thermal_profile(chip_name: str) -> ThermalProfile:
    """Look up the thermal profile for a chip by name."""
    if chip_name in THERMAL_PROFILES:
        return THERMAL_PROFILES[chip_name]
    raise KeyError(f"no thermal profile registered for '{chip_name}'")
