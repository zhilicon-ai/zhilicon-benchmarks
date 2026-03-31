# Contributing to zhilicon-benchmarks

## Ways to Contribute

- **New benchmark workloads** — model types or access patterns not yet covered
- **Bug fixes** — incorrect measurements, broken scripts, stale configs
- **Methodology improvements** — better statistical rigor, new measurement dimensions
- **Documentation** — clearer reproduction instructions

## Important Rules for Benchmark Contributions

1. **Results must be reproducible** — include full hardware/software config
2. **No unreleased silicon results** — only B0 and later revisions with published specs
3. **Methodology changes require discussion first** — open an issue before submitting a PR that changes how measurements are taken
4. **Sign-off required** — all commits must include `Signed-off-by` (use `git commit -s`)

## Pull Request Guidelines

- One benchmark or fix per PR
- Include before/after results if changing measurement methodology
- New workloads must have a `README.md` explaining what is measured and why

## Review Process

PRs are reviewed by `@zhilicon-ai/ml-systems` and `@zhilicon-ai/sdk`. Results PRs additionally require `@zhilicon-ai/program-management` approval before merge.
