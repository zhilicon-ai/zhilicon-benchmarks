# zhilicon-benchmarks

> Official benchmark suite for the Zhilicon AI Chip — methodology, scripts, and published results.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

This repository contains the complete benchmark methodology, test harnesses, and published performance results for Zhilicon silicon. It is the authoritative source for performance claims and enables independent reproduction of results.

**Published results** are in [`results/`](results/). **Benchmark scripts** are in [`benchmarks/`](benchmarks/). The methodology and measurement protocol are in [`docs/methodology.md`](docs/methodology.md).

---

## Benchmark Suite

| Suite | What It Measures |
|-------|-----------------|
| `inference-throughput` | Peak tokens/sec and images/sec across batch sizes |
| `inference-latency` | P50/P95/P99 latency at target throughput levels |
| `memory-bandwidth` | HBM3 read/write bandwidth at sustained load |
| `power-perf` | TOPS/W and latency at multiple TDP points |
| `model-zoo` | End-to-end accuracy and performance across 20+ standard models |
| `multi-chip-scaling` | Throughput scaling efficiency: 1×, 2×, 4×, 8× chips |

---

## Repository Structure

```
zhilicon-benchmarks/
├── benchmarks/
│   ├── inference-throughput/
│   ├── inference-latency/
│   ├── memory-bandwidth/
│   ├── power-perf/
│   ├── model-zoo/
│   └── multi-chip-scaling/
├── results/
│   ├── B0/                    # B0 silicon results (by date)
│   └── simulator/             # Pre-silicon simulator results
├── models/                    # Model configs and ONNX graph specs
├── docs/
│   ├── methodology.md         # Full measurement protocol
│   ├── reproducibility.md     # How to reproduce published results
│   └── comparison-notes.md    # Comparison methodology and caveats
└── tools/
    ├── harness.py             # Benchmark harness runner
    └── report_gen.py          # Results report generator
```

---

## Quick Start

```bash
git clone https://github.com/zhilicon-ai/zhilicon-benchmarks
cd zhilicon-benchmarks
pip install -r requirements.txt

# Run a single benchmark
python tools/harness.py --suite inference-throughput --model llama-3-8b --device zhi1

# Run full model-zoo sweep
python tools/harness.py --suite model-zoo --device zhi1 --output results/my-run/
```

---

## Published Results

Results are published in [`results/`](results/) as structured JSON with a human-readable summary.

| Chip | Silicon Rev | Date | Report |
|------|-------------|------|--------|
| ZHI-1 | B0 | TBD | [results/B0/](results/B0/) |

---

## Reproducing Results

See [`docs/reproducibility.md`](docs/reproducibility.md) for:
- Exact hardware and software configuration
- Thermal and power measurement setup
- Statistical methodology (number of runs, warm-up, outlier handling)
- Known sources of variance

---

## Methodology

All Zhilicon benchmarks follow the principles in [`docs/methodology.md`](docs/methodology.md):

- **Measured at steady state** — no cold-start artifacts
- **Power measured at the wall** — not from chip PMU alone
- **Latency is end-to-end** — includes host-device transfer
- **All precisions reported** — FP16, BF16, INT8, FP8
- **Comparison caveats documented** — different chips have different optimization surfaces

---

## Contributing

We welcome bug reports, methodology improvements, and new benchmark workloads. See [CONTRIBUTING.md](CONTRIBUTING.md).

**Important:** Do not submit results from unreleased hardware or software versions. All contributed results must be reproducible.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

Model weights are not distributed in this repo. See individual model licenses.
