# Reproducing Published Results

This guide walks you through reproducing any result published in this repository. Follow these steps in order. If your result differs from the published number, consult the "What to do if results differ" section at the end.

---

## Step 1: Hardware Setup Checklist

Before running any benchmark, verify your hardware matches the configuration in the result file's `hardware` block.

### Required hardware

- [ ] ZHI-1 B0 evaluation board (or later revision — note that results may differ)
- [ ] Host system: x86-64, PCIe Gen 5 x16 slot available
- [ ] Host RAM: 64 GB minimum (256 GB recommended for LLM benchmarks)
- [ ] Active cooling: chip junction temperature must stay below 85°C throughout the run

### BIOS and system settings

Apply these settings before benchmarking. Incorrect settings are the most common source of latency discrepancies.

| Setting | Required Value | Where to Set |
|---------|---------------|--------------|
| CPU governor | performance | `echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| CPU C-states | disabled | BIOS: Power Management → C-States → Disabled |
| PCIe ASPM | disabled | BIOS: PCIe → ASPM → Disabled |
| NUMA balancing | disabled | `echo 0 > /proc/sys/kernel/numa_balancing` |
| Transparent hugepages | madvise | `echo madvise > /sys/kernel/mm/transparent_hugepage/enabled` |
| Hyperthreading | enabled | BIOS: leave default (enabled) |

### Power measurement setup (for power/TOPS-W results)

- Connect a calibrated power meter inline with the board 12V rail
- Connect a second meter to the PCIe slot (12V + 3.3V pins)
- Verify meter calibration against a known reference load before measurement
- Meters must sample at ≥ 1 Hz and export timestamped readings

Power results cannot be reproduced without external metering equipment. If you are reproducing throughput/latency only, power measurement hardware is not required.

---

## Step 2: Software Installation

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-dev python3-pip git

# Clone this repository
git clone https://github.com/zhilicon-ai/zhilicon-benchmarks
cd zhilicon-benchmarks

# Install Python dependencies
pip3.11 install -r requirements.txt

# Install Zhilicon SDK
# Use the SDK version recorded in the result file's software.sdk_version field
pip3.11 install zhilicon-sdk==1.0.0 \
  --index-url https://pypi.zhilicon.ai/simple/

# Verify SDK installation
python3.11 -c "import zhilicon; print(zhilicon.__version__)"

# Install the ZHI-1 kernel driver (hardware only — skip for simulator)
sudo dpkg -i /path/to/zhilicon-driver-1.0.0.deb
sudo modprobe zhilicon
```

### Verify hardware is visible (hardware only)

```bash
python3.11 -c "
import zhilicon
devices = zhilicon.list_devices()
for d in devices:
    print(f'{d.index}: {d.name} rev={d.revision} mem={d.memory_gb}GB')
"
# Expected: 0: ZHI-1 rev=B0 mem=32GB
```

---

## Step 3: Running a Specific Benchmark

Each published result file is in `results/B0/<date>/`. To reproduce a specific result:

### Find the result file

```bash
# List available B0 results
ls results/B0/

# Example: 2026-04-01/resnet50-fp16-batch64.json
cat results/B0/2026-04-01/resnet50-fp16-batch64.json | python3.11 -m json.tool
```

### Extract the benchmark parameters

From the result file, note:
- `benchmark_suite` — which suite to run
- `model` — which model
- `precision` — fp32/fp16/bf16/int8/fp8
- `batch_size` — batch size used

### Run the benchmark

```bash
# Hardware run
python3.11 tools/harness.py \
  --suite inference-throughput \
  --model resnet50 \
  --precision fp16 \
  --batch-size 64 \
  --device zhi1 \
  --runs 5 \
  --output results/my-run/

# Simulator run (no hardware required — results will differ from published)
export ZHILICON_DEVICE=simulator
python3.11 tools/harness.py \
  --suite inference-throughput \
  --model resnet50 \
  --precision fp16 \
  --batch-size 64 \
  --device simulator \
  --runs 5 \
  --output results/my-run/
```

The harness will:
1. Run the warmup protocol (30 seconds minimum, up to 5 minutes)
2. Measure for 60 seconds at steady state
3. Repeat for the specified number of runs
4. Apply Grubbs outlier rejection
5. Write a result JSON to `results/my-run/`

---

## Step 4: Comparing Your Result to the Published Result

```bash
# Validate your result file structure
python3.11 scripts/validate_results.py results/my-run/

# Generate a comparison report
python3.11 tools/report_gen.py \
  --result results/my-run/ \
  --compare results/B0/2026-04-01/resnet50-fp16-batch64.json
```

The comparison report will show:
- Your throughput vs. published throughput (% difference)
- Your latency percentiles vs. published
- Your power vs. published (if measured)
- Any configuration differences detected

### Acceptable variance

| Metric | Acceptable Difference |
|--------|-----------------------|
| Throughput | ±5% from published mean |
| Latency P50 | ±10% |
| Latency P99 | ±15% |
| Power | ±8% (due to board-to-board variation) |

Differences within these ranges are normal due to board-to-board variation, ambient temperature, and PCIe topology differences. Differences outside these ranges require investigation.

---

## Step 5: What to Do if Your Results Differ

Work through these checks in order.

### Check 1: Software version mismatch

```bash
python3.11 -c "import zhilicon; print(zhilicon.__version__)"
```

Compare to `software.sdk_version` in the result file. A newer SDK may produce higher throughput (compiler improvements). An older SDK may produce lower throughput. If versions differ, this is the most likely cause.

### Check 2: System settings

Verify all BIOS and OS settings from the hardware setup checklist. The most common culprits:

- CPU C-states enabled → adds 1–5ms latency jitter
- PCIe ASPM enabled → reduces PCIe throughput by 10–30%
- CPU governor not set to performance → throughput variance increases

### Check 3: Thermal throttling

Check chip temperature during the measurement window:

```bash
python3.11 -c "
import zhilicon
d = zhilicon.Device(0)
print(f'Chip temp: {d.temperature_c}°C')
"
```

If temperature exceeds 85°C, the chip throttles. Ensure active cooling is working and ambient temperature is below 25°C.

### Check 4: Other PCIe devices competing for bandwidth

Run `lspci` and ensure no other high-bandwidth PCIe devices (NVMe drives, network cards) are active during the measurement. In particular, NVMe drives on the same PCIe root complex can reduce ZHI-1 throughput by up to 15% if they are doing concurrent I/O.

### Check 5: File a reproducibility issue

If you have verified all of the above and your results still differ by more than the acceptable range, please [file a GitHub issue](https://github.com/zhilicon-ai/zhilicon-benchmarks/issues/new/choose) with:

- Your result JSON file
- Output of `python3.11 scripts/validate_results.py results/my-run/`
- Your system hardware configuration
- The published result file you are comparing against
- Description of what you tried to resolve the discrepancy

Reproducibility issues are treated as high-priority bugs.
