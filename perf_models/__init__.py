"""
Zhilicon Performance Modeling Framework
=========================================

A first-principles analytical performance model for the 5-chip Zhilicon
portfolio (Sentinel-1, Horizon-1, Discovery-1, Nexus-1, Prometheus). Used
to project achievable PPA (power / performance / area) under different
workload mixes before silicon arrives, and to validate post-silicon
measurements against the architecture's claimed numbers.

Layers
------

1. **chips/**       : per-chip top-level models (parameters, clocks, power
                      rails, fabric topology).
2. **kernels/**     : per-operator analytical roofline models (matmul,
                      conv, attention, FFT, NTT, AES, ...).
3. **power/**       : dynamic + leakage power decomposition with DVFS
                      operating-point sweeps.
4. **thermal/**     : 1D thermal-network model with package theta-JA.
5. **workloads/**   : reference workloads composing kernels into full
                      end-to-end traces (LLaMA-7B inference, ResNet-50
                      training, AES-XTS, Kyber keygen, OFDM-256 modem,
                      DICOM-3D-CNN, ...).

Conventions
-----------

* All times are in **nanoseconds** unless otherwise stated.
* All energy values are in **picojoules** (pJ) unless stated.
* All bandwidth values are in **GB/s** (decimal, not GiB/s).
* All frequencies are in **GHz**.
* All voltages are in **millivolts** (mV) to keep arithmetic in integers
  where possible.
* All temperatures are in **degrees Celsius**.

Reference: ZH-PERF-MODEL-001
"""

__version__ = "0.1.0"

from .chips import (
    Sentinel1Model,
    Horizon1Model,
    Discovery1Model,
    Nexus1Model,
    PrometheusModel,
)

__all__ = [
    "Sentinel1Model",
    "Horizon1Model",
    "Discovery1Model",
    "Nexus1Model",
    "PrometheusModel",
]
