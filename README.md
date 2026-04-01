# zhilicon-benchmarks

[![CI](https://github.com/zhilicon-ai/zhilicon-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/zhilicon-ai/zhilicon-benchmarks/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Methodology](https://img.shields.io/badge/methodology-v1.0-brightgreen)](docs/METHODOLOGY.md)

> Official benchmark suite, methodology, and published performance results for the Zhilicon ZHI-1 AI chip. Reproducible measurements. Transparent methodology. No cherry-picking.

---

## Why This Repository

Performance benchmarks for AI accelerators are frequently misleading: peak vs. sustained throughput, wall-clock vs. chip-only power, batch-1 vs. optimal-batch latency. This repository exists to provide a single, reproducible source of truth for ZHI-1 performance — with the full methodology published so you can verify every number yourself.

---

## Results at a Glance

> Results from ZHI-1 B0 silicon. Full configuration in [`results/B0/hardware-config.json`](results/B0/).

| Model | Task | Precision | Throughput | Latency (P99) | Power | TOPS/W |
|-------|------|-----------|-----------|--------------|-------|--------|
| ResNet-50 | Image classification | INT8 | 18,400 img/s | 2.1ms | 120W | — |
| LLaMA-3-8B | LLM decode | FP16 | 3,200 tok/s | 0.8ms/tok | 180W | — |
| LLaMA-3-70B | LLM decode (4-chip) | FP16 | 890 tok/s | 2.1ms/tok | 640W | — |
| BERT-large | Encoding | FP16 | 12,000 seq/s | 3.4ms | 165W | — |
| YOLOv8-L | Object detection | INT8 | 4,200 img/s | 1.2ms | 135W | — |

*All numbers at steady state, wall-clock power, end-to-end (host to host). See [methodology](docs/METHODOLOGY.md) for full measurement protocol.*

---

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
│   ├── B0/                    # ZHI-1 B0 silicon results (dated subdirectories)
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

Measures throughput scaling efficiency with 1, 2, 4, and 8 ZHI-1 chips. Reports scaling efficiency (ideal = N× linear).

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
