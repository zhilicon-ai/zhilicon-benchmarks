# Benchmark Methodology v1.0

This document defines the complete measurement protocol for all Zhilicon benchmark results. Every published number in this repository was produced using the protocol defined here. The methodology version is recorded in each result JSON file so that results can always be traced back to the protocol used.

---

## Principles

These benchmarks follow five core principles:

1. **Reproducibility** — every result can be reproduced by a third party with the same hardware and software configuration
2. **Transparency** — measurement protocol is fully published; nothing is hidden
3. **Conservative measurement** — when in doubt, we report the harder number (sustained vs. burst, end-to-end vs. chip-only)
4. **Statistical rigor** — multiple runs, confidence intervals, outlier rejection
5. **Honest comparisons** — we document sources of variance and note where comparisons are and are not fair

---

## Hardware Configuration

All published results use the following configuration unless otherwise noted in the result file's `hardware` field.

| Component | Specification |
|-----------|--------------|
| ZHI-1 revision | B0 (first silicon) |
| Host CPU | Intel Xeon Gold 6338 @ 2.0 GHz (2× socket) |
| Host RAM | 256 GB DDR5 ECC |
| PCIe | Gen 5 x16 |
| Board | Zhilicon ZHI-1 B0 evaluation board rev 1 |
| Cooling | Active cooling; chip junction temperature < 85°C throughout measurement window |
| Power supply | 1200W ATX, calibrated against reference meter |

---

## Software Configuration

| Component | Version |
|-----------|---------|
| Host OS | Ubuntu 22.04 LTS |
| Kernel | 5.15.0-91-generic or later |
| Zhilicon SDK | Recorded in each result file's `software.sdk_version` field |
| Python | 3.11 |
| ONNX Runtime | Latest stable at time of measurement (recorded in result file) |
| Compiler optimization level | `--opt-level 3` (maximum) |

If you are reproducing results, your software configuration must match or be newer. Results with older SDK versions may differ due to compiler improvements.

---

## Warmup Protocol

AI accelerators frequently show higher throughput immediately after start (burst mode) than at steady state, due to thermal throttling, memory controller state changes, and cache effects. All Zhilicon benchmarks use the following warmup to ensure steady-state measurement:

1. Run the workload continuously at full load
2. Monitor throughput using a 10-second sliding window
3. Continue warmup until the coefficient of variation (CV = σ/μ) of the sliding window drops below 0.01 (1%)
4. **Minimum warmup:** 30 seconds regardless of CV — this catches cases where variance appears low but hasn't reached true steady state
5. **Maximum warmup:** 5 minutes — if steady state is not reached within 5 minutes, this is flagged as `"warmup_converged": false` in the result and the measurement proceeds with a note

Warmup duration is recorded in the result file's `results.warmup_time_seconds` field.

---

## Measurement Window

After warmup completes, measure over a **60-second steady-state window**. During this window, record:

- Total operations completed
- Throughput = `total_ops / 60.0`
- Per-operation timestamps for latency distribution computation
- Power meter readings (sampled at ≥ 1 Hz)
- Chip junction temperature at start and end of window

The measurement window starts when the host submits the first operation and ends exactly 60 seconds later. Operations in flight at the 60-second mark are not counted.

---

## Latency Measurement

Latency is measured as the time elapsed from when the host submits a request to when the result is fully available in host memory. This definition deliberately includes:

- API call dispatch overhead
- Device queue depth waiting time (if any)
- Kernel execution time on chip
- Result DMA transfer back to host memory
- Any host-side post-processing performed by the SDK

Latency is measured at the following load levels:

| Load Level | Description |
|------------|-------------|
| Batch=1, serial | One request at a time, no concurrency |
| 50% of peak throughput | Realistic loaded-system latency |
| 90% of peak throughput | Near-saturation latency |

Latency percentiles (P50, P90, P95, P99) are computed over a minimum of **1,000 individual operations** within the measurement window.

---

## Power Measurement

Power is measured at the hardware level, not from on-chip estimates. The measurement setup:

- A calibrated power meter on the 12V board power rail
- PCIe slot power measured separately at the 12V and 3.3V pins
- Total power = `board_12V_watts + PCIe_12V_watts + PCIe_3.3V_watts × (3.3 / 12)`

Power is sampled at a minimum of 1 Hz throughout the 60-second measurement window. The reported value is the mean over the measurement window (not peak, not minimum).

**Why we do not use on-chip PMU readings:** PMU power estimates are calibrated against a specific reference workload and can deviate by 15–30% from actual power at the wall, depending on workload characteristics. For comparability with other AI accelerators (which typically report wall power), we use external measurement.

Power measurement setup is documented in the `hardware` block of each result file.

---

## Statistical Methodology

### Number of Runs

Minimum **5 independent runs** per measurement point. Each run constitutes a complete sequence: process start → warmup → 60-second measurement window → process exit. The process is fully restarted between runs (no state sharing).

### Outlier Rejection

Apply the **Grubbs test** at 95% confidence to each set of runs. Identified outliers are flagged in the result file (`results.outliers` array) and excluded from mean/stddev computation, but not silently discarded. The raw per-run data is included in the result file.

### Reported Values

| Metric | Reported As |
|--------|-------------|
| Throughput | Mean ± 1 standard deviation across runs |
| Latency P50/P95/P99 | Pooled distribution across all runs |
| Power | Mean across all measurement windows |

### Run-to-Run Variance Targets

| Metric | Target CV |
|--------|-----------|
| Throughput | < 2% |
| Latency P99 | < 5% |
| Power | < 3% |

If variance exceeds these targets, additional runs are performed (up to 10 total). If variance still exceeds targets, a `"high_variance": true` flag is set in the result with an explanation.

---

## Multi-Chip Configuration

For multi-chip scaling benchmarks (2×, 4×, 8× chips):

- All chips are on the same host platform
- PCIe switch topology is documented in the `hardware` block of the result file
- Workload partitioning strategy is documented (pipeline parallel, data parallel, or tensor parallel)
- Power is measured for all chips simultaneously using the board-level rail
- Scaling efficiency = `(N-chip throughput) / (N × 1-chip throughput)` — 100% means perfect linear scaling

---

## Accuracy Measurement (model-zoo suite)

For model-zoo benchmarks that report accuracy:

- Accuracy is measured on the standard validation dataset for each model (ImageNet val for image classifiers, etc.)
- Reference accuracy is measured on CPU using the same model weights and ONNX graph
- Reported accuracy delta = `ZHI-1 accuracy − CPU reference accuracy`
- A delta of 0.0% means bit-exact output; a negative delta indicates numerical degradation from precision reduction

---

## Compiler Optimization Settings

All published results use maximum compiler optimization (`--opt-level 3`). This enables:

- Operator fusion
- Memory layout optimization for HBM3 access patterns
- Quantization-aware kernel selection (for INT8/FP8)
- Graph-level scheduling optimization

Lower optimization levels produce different (typically lower) throughput. If you use a lower optimization level, note it in your result file.

---

## Comparison Guidance

When comparing ZHI-1 results to results from other hardware vendors:

1. **Batch size** — always compare at the same batch size, or compare the full throughput-latency curve. A chip optimized for large batches will appear faster at high batch sizes but slower at batch=1.
2. **Precision** — INT8 on one chip vs. FP16 on another is not a valid comparison. Always compare at the same precision, and verify the other chip's INT8 path does not sacrifice accuracy.
3. **System boundary** — always use end-to-end, host-to-host measurements for both chips. Chip-only numbers exclude host overhead and make the chip appear faster than it is in a real system.
4. **Thermal conditions** — sustained numbers (60s window) vs. burst numbers (5s or shorter window) differ significantly for thermally limited devices. Verify which measurement window was used.
5. **Software maturity** — early silicon often has immature compiler and runtime. ZHI-1 B0 results will improve with later SDK releases. This is true of all first-silicon benchmarks.
6. **Power envelope** — some chips report TDP (thermal design power) rather than actual measured power. Actual power can be 20–40% lower than TDP for many workloads.

---

## Result File Format

Each result run produces a JSON file conforming to the following schema. All fields are required unless marked optional.

```json
{
  "hardware": {
    "chip": "ZHI-1",
    "revision": "B0",
    "count": 1,
    "board": "zhilicon-eval-b0-v1",
    "host_cpu": "Intel Xeon Gold 6338",
    "host_ram_gb": 256,
    "pcie_gen": 5,
    "cooling": "active",
    "ambient_temp_c": 22
  },
  "software": {
    "sdk_version": "1.0.0",
    "os": "Ubuntu 22.04",
    "kernel": "5.15.0-91-generic",
    "python": "3.11.7",
    "onnxruntime": "1.16.3",
    "compiler_opt_level": 3
  },
  "benchmark_suite": "inference-throughput",
  "model": "resnet50",
  "precision": "fp16",
  "batch_size": 64,
  "methodology_version": "1.0",
  "date": "2026-04-01",
  "results": {
    "throughput_mean": 18400.0,
    "throughput_stddev": 185.0,
    "throughput_unit": "images/sec",
    "latency_p50_ms": 1.8,
    "latency_p90_ms": 1.9,
    "latency_p95_ms": 2.0,
    "latency_p99_ms": 2.1,
    "power_mean_watts": 120.0,
    "power_stddev_watts": 2.1,
    "chip_temp_start_c": 72,
    "chip_temp_end_c": 78,
    "runs": 5,
    "outliers": [],
    "warmup_time_seconds": 45,
    "warmup_converged": true,
    "high_variance": false,
    "per_run_throughput": [18210.0, 18450.0, 18390.0, 18520.0, 18430.0]
  }
}
```

---

## Known Limitations and Variance Sources

- **B0 silicon maturity:** These are first-silicon results. The compiler, runtime, and memory controller firmware will improve in subsequent releases. Performance numbers will change.
- **Thermal variance:** Ambient temperature ±5°C can affect sustained throughput by 2–3% for thermally limited workloads. All published results are measured at 22°C ± 1°C ambient.
- **Memory controller initialization:** The first benchmark run after a cold system start may show 5–10% lower throughput than subsequent runs due to memory controller training state. The warmup protocol mitigates this, but it is a known source of run-0 variance.
- **PCIe bandwidth sharing:** On platforms with multiple PCIe devices, bandwidth sharing with other devices (e.g., NVMe storage, NICs) can affect host-to-device transfer latency. Published results are measured with no other PCIe devices active.
- **BIOS/firmware settings:** NUMA balancing, CPU C-states, and PCIe ASPM settings can affect latency by 5–15%. Published results use: NUMA balancing disabled, C-states disabled (performance governor), PCIe ASPM disabled.

---

## Methodology Version History

| Version | Date | Summary of Changes |
|---------|------|--------------------|
| 1.0 | 2026-04-01 | Initial published methodology |
