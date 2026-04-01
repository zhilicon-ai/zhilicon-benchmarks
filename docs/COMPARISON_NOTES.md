# Comparison Notes

This document provides guidance for making fair comparisons between ZHI-1 results in this repository and results published for other AI accelerators.

Comparisons are useful. Misleading comparisons waste everyone's time. This document exists to help you compare honestly.

---

## The Five Most Common Ways Benchmark Comparisons Go Wrong

### 1. Peak vs. sustained throughput

Many vendors publish burst throughput measured over a 1–5 second window before thermal throttling begins. Zhilicon publishes sustained throughput measured over a 60-second steady-state window after warmup. These numbers are not comparable.

**How to check:** Look for the measurement window duration in the other vendor's methodology. If it is not published, the number is probably burst throughput.

### 2. Chip-only vs. end-to-end

Chip-only throughput excludes host-to-device data transfer and device-to-host result transfer. For inference workloads, host-device transfers can represent 10–40% of end-to-end latency depending on model input size and PCIe bandwidth.

Zhilicon publishes end-to-end, host-to-host numbers. If the comparison target publishes chip-only numbers, the comparison will favor the comparison target unfairly.

**How to check:** Look for language like "device compute time only" or "excluding data transfer" in the other vendor's methodology.

### 3. Batch size mismatch

Throughput at batch=64 vs. throughput at batch=1 can differ by 10–100× depending on the chip architecture. Always compare at the same batch size, or compare the full throughput-latency curve rather than a single operating point.

### 4. Precision mismatch

INT8 throughput is typically 2–4× higher than FP16 throughput on the same chip. Always compare at the same precision. If the other chip does not support a given precision, note that the comparison is between two different numeric regimes.

### 5. Software maturity differences

First-silicon results for any chip are typically 20–40% below what the same chip achieves with a mature compiler and runtime (6–18 months after initial release). ZHI-1 B0 results reflect early compiler maturity. They will improve.

---

## Fair Comparison Checklist

Before publishing a comparison that includes ZHI-1 numbers from this repository, verify:

- [ ] Same measurement window duration (Zhilicon uses 60 seconds)
- [ ] Same system boundary (Zhilicon uses host-to-host)
- [ ] Same batch size
- [ ] Same model and precision
- [ ] Power measured the same way (Zhilicon uses external meter on board rail)
- [ ] Same model weights (different quantization approaches produce different accuracy/performance tradeoffs)
- [ ] Software versions documented for both chips

If any of these do not match, document the mismatch explicitly rather than omitting it.

---

## Questions and Disputes

If you believe a comparison using Zhilicon numbers is misleading, please [open an issue](https://github.com/zhilicon-ai/zhilicon-benchmarks/issues/new) with:

- A link to the comparison
- Which specific numbers you believe are being misrepresented
- What you believe the correct comparison should show

We take benchmark integrity seriously and will respond to all such issues.
