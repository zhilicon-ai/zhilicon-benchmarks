# Contributing to zhilicon-benchmarks

Benchmark integrity depends on community vigilance. Contributions that improve methodology rigor, add new workloads, fix broken scripts, or improve documentation are all welcome.

---

## Ways to Contribute

- **New benchmark workloads** — model types or access patterns not yet covered by the existing suites
- **Bug fixes** — incorrect measurements, broken scripts, stale configurations
- **Methodology improvements** — better statistical rigor, additional measurement dimensions, improved warmup detection
- **Documentation** — clearer reproduction instructions, better comparison notes
- **Results validation** — reproducing published results on independent hardware and reporting deviations

---

## Results Contribution Requirements

Any PR that adds or modifies benchmark results **must** include the following. PRs missing these fields will not be merged.

### 1. Hardware Configuration

A `hardware-config.json` file (or inline in the result JSON) with:

```json
{
  "chip": "ZHI-1",
  "revision": "B0",
  "chip_count": 1,
  "board": "zhilicon-eval-b0-v1",
  "host_cpu": "Intel Xeon Gold 6338 @ 2.0GHz",
  "host_ram_gb": 256,
  "pcie_gen": 5,
  "cooling": "active",
  "ambient_temp_c": 22
}
```

### 2. Software Configuration

```json
{
  "os": "Ubuntu 22.04 LTS",
  "kernel": "5.15.0-91-generic",
  "sdk_version": "1.0.0",
  "python": "3.11.7",
  "onnxruntime": "1.16.3",
  "methodology_version": "1.0"
}
```

### 3. Methodology Version

The `methodology_version` field in every result file must reference the version of `docs/METHODOLOGY.md` that was followed. If you followed a version other than the current one, explain why.

### 4. Reproducibility Evidence

Include a brief description of how you verified reproducibility: number of independent runs, run-to-run variance observed, and any deviations from standard methodology.

---

## New Benchmark Workload Requirements

New benchmark workloads added to the `benchmarks/` directory must satisfy all of the following:

1. **CI-runnable validator** — a `validate.py` (or similar) script that runs on the simulator without real hardware, verifies the benchmark output schema, and exits 0 on success
2. **README** — a `README.md` in the workload directory explaining: what is measured, why it matters, expected output schema, and any known measurement pitfalls
3. **Schema compliance** — output JSON must conform to the result schema defined in `docs/METHODOLOGY.md`
4. **Methodology alignment** — the workload must follow warmup, measurement window, power, and statistical methodology defined in `docs/METHODOLOGY.md`

---

## Methodology Change Process

Changes to `docs/METHODOLOGY.md` affect the validity of all previously published results. The process is:

1. Open a GitHub Discussion issue with the label `methodology-change` describing the proposed change and its rationale
2. Allow at least 7 days for community and team comment
3. If approved, open a PR — methodology version number must be incremented
4. All existing results must note which methodology version they used

Do not open a methodology-change PR without a prior discussion issue. Such PRs will be closed without review.

---

## DCO Sign-Off

All commits must include a Developer Certificate of Origin sign-off. Add it with:

```bash
git commit -s -m "bench: add YOLOv8-XL throughput benchmark"
```

This adds a `Signed-off-by: Your Name <email@example.com>` trailer to the commit. By signing off you certify that you wrote the contribution or have the right to contribute it under the Apache 2.0 license.

---

## Pull Request Guidelines

- **One benchmark, fix, or topic per PR** — keep PRs focused for easier review
- **New workloads**: include the CI validator, README, and at least one example result (simulator output is acceptable)
- **Results PRs**: include all required fields listed above
- **Methodology changes**: reference the prior discussion issue
- **Tests**: ensure `python scripts/validate_results.py` passes on all result files in your PR

---

## PR Review Process

| PR Type | Required Reviewers |
|---------|-------------------|
| Documentation, scripts, tooling | `@zhilicon-ai/ml-systems` |
| New benchmark workloads | `@zhilicon-ai/ml-systems` + `@zhilicon-ai/sdk` |
| Published results | `@zhilicon-ai/ml-systems` + `@zhilicon-ai/program-management` |
| Methodology changes | `@zhilicon-ai/ml-systems` + `@zhilicon-ai/program-management` + community discussion |

Results PRs require **both** ml-systems and program-management approval before merge. This two-party sign-off ensures numbers are technically correct and accurately represent the measurement conditions.

---

## What Not to Submit

- Results from unreleased hardware revisions not yet publicly announced
- Results from pre-release SDK versions unless clearly marked as pre-release
- Benchmark workloads that cannot be validated in CI
- Anything received under NDA or marked confidential
- Model weights (reference model sources in the workload README instead)

---

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful in issues, PRs, and discussions. Methodological disagreements are healthy — personal attacks are not.
