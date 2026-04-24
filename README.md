<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/zhilicon-ai/.github/main/profile/assets/zhilicon-logo-dark.png" width="320">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/zhilicon-ai/.github/main/profile/assets/zhilicon-logo-light.png" width="320">
  <img alt="Zhilicon" src="https://raw.githubusercontent.com/zhilicon-ai/.github/main/profile/assets/zhilicon-logo-light.png" width="320">
</picture>

# Zhilicon Benchmarks

### Reproducible performance benchmark suite for the Zhilicon five-chip portfolio. Activates at first-silicon bring-up (Sentinel-1 Q4 2026).

[![CI](https://github.com/zhilicon-ai/zhilicon-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/zhilicon-ai/zhilicon-benchmarks/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/zhilicon-ai/zhilicon-benchmarks?include_prereleases&sort=semver&color=0d1117&label=release)](https://github.com/zhilicon-ai/zhilicon-benchmarks/releases/latest)
[![Last Commit](https://img.shields.io/github/last-commit/zhilicon-ai/zhilicon-benchmarks?color=0d1117&label=last%20commit)](https://github.com/zhilicon-ai/zhilicon-benchmarks/commits/main)
[![Portfolio](https://img.shields.io/badge/Zhilicon-v0.2.0-0d1117)](https://github.com/zhilicon-ai)

[![Methodology](https://img.shields.io/badge/methodology-v1.0-0d1117)](docs/METHODOLOGY.md)
[![Status](https://img.shields.io/badge/status-pre_silicon-yellow)](https://github.com/zhilicon-ai/zhilicon-benchmarks)

</div>

---

<p align="center">
  <a href="https://github.com/zhilicon-ai"><strong>Portfolio</strong></a>&nbsp;·&nbsp;
  <a href="https://github.com/zhilicon-ai/zhilicon-sdk"><strong>SDK</strong></a><sup>🔒</sup>&nbsp;·&nbsp;
  <a href="https://github.com/zhilicon-ai/zhilicon-sdk-examples"><strong>Examples</strong></a>&nbsp;·&nbsp;
  <a href="https://github.com/zhilicon-ai/zhilicon-developer-docs"><strong>Developer Docs</strong></a>&nbsp;·&nbsp;
  <a href="https://github.com/zhilicon-ai/zhilicon-benchmarks/releases"><strong>Releases</strong></a>
</p>

---

## Why This Repository

Performance benchmarks for AI accelerators are frequently misleading: peak vs. sustained throughput, wall-clock vs. chip-only power, batch-1 vs. optimal-batch latency. This repository exists to provide a single, reproducible source of truth for Zhilicon portfolio performance — with the full methodology published so you can verify every number yourself.

---

## Results at a Glance

> **Pre-silicon status.** This repository activates at first-silicon bring-up.
> Sentinel-1 is scheduled Q4 2026; Discovery-1 and Prometheus Q3 2026; Nexus-1 Rev A Q4 2026;
> Horizon-1 Q1 2027. Once silicon arrives, this section will publish measured results with
> full reproducibility methodology for every number.
>
> Until then, the repository ships the **methodology framework** (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md))
> so that when silicon arrives, the first published result follows the same measurement
> contract every subsequent result will.


## Quick Start — Reproduce a Result

```bash
# Clone
git clone https://github.com/zhilicon-ai/zhilicon-benchmarks
cd zhilicon-benchmarks

# Install dependencies
pip install -r requirements.txt

# Run a single benchmark on the simulator
# (Hardware access required for production numbers)
export ZHILICON_DEVICE=simulator
python tools/harness.py \
  --suite inference-throughput \
  --model resnet50 \
  --precision fp16 \
  --device simulator \
  --output results/my-run/

# View results
python tools/report_gen.py results/my-run/
```

---

## Repository Structure

```
zhilicon-benchmarks/
├── benchmarks/
│   ├── inference-throughput/  # Peak throughput across batch sizes and precisions
│   ├── inference-latency/     # P50/P95/P99 latency at target throughput levels
│   ├── memory-bandwidth/      # HBM3 sustained read/write bandwidth
│   ├── power-perf/            # TOPS/W and efficiency across TDP points
│   ├── model-zoo/             # End-to-end accuracy + performance (20+ models)
│   └── multi-chip-scaling/    # Throughput scaling: 1×, 2×, 4×, 8× chips
├── results/
│   ├── B0/                    # Zhilicon silicon (pre-tape-out) results (dated subdirectories)
│   └── simulator/             # Functional simulator results (not performance)
├── models/
│   └── configs/               # Model configurations and graph specs
├── docs/
│   ├── METHODOLOGY.md         # Full measurement protocol
│   ├── REPRODUCIBILITY.md     # Step-by-step reproduction guide
│   └── COMPARISON_NOTES.md    # Fair comparison guide
├── tools/
│   ├── harness.py             # Benchmark harness runner
│   ├── validate_results.py    # Result file schema validator
│   └── report_gen.py          # Results report generator
└── scripts/
    └── validate_results.py    # CI validation script
```

---

## Benchmark Suites

### inference-throughput

Measures sustained tokens/second or images/second at steady state. Varies batch size from 1 to maximum. Reports: peak throughput, optimal batch size, throughput at batch=1.

### inference-latency

Measures P50/P95/P99 latency at 50% peak throughput load point. Reports percentile distribution. Tests: synchronous and asynchronous dispatch.

### memory-bandwidth

Measures HBM3 sustained read/write bandwidth using synthetic memory-bound kernels. Reports peak and sustained bandwidth.

### power-perf

Measures TOPS/W at multiple TDP operating points. Power measured at the wall with a calibrated power meter. Reports efficiency curve.

### model-zoo

Runs 20+ standard ML models end-to-end. Reports: throughput, latency, accuracy (vs. reference CPU run). Models include: ResNet variants, EfficientNet, BERT, LLaMA, T5, YOLOv8, Stable Diffusion (text encoder).

### multi-chip-scaling

Measures throughput scaling efficiency with 1, 2, 4, and 8 Zhilicon portfolio chips. Reports scaling efficiency (ideal = N× linear).

---

## Measurement Methodology

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full protocol. Key principles:

1. **Steady-state measurement** — warmup until throughput stabilizes (CV < 1%), then measure over a 60-second window
2. **Wall-clock power** — measured at the PCIe slot power pins + board 12V rail with a calibrated meter, not from PMU estimates
3. **End-to-end latency** — from host API call return to result available in host memory
4. **Multiple runs** — minimum 5 runs; outlier rejection using Grubbs test; report mean ± 1σ
5. **All precisions** — FP32, FP16, BF16, INT8, FP8 reported where hardware supports

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Key requirements:

- Results must be reproducible from your hardware config + software config
- New benchmark workloads must have a CI-runnable validator
- Methodology changes require a discussion issue before PR

---

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights are not distributed here.
