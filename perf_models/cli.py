"""Command-line interface to the perf-model framework.

Usage examples:

    python -m perf_models summary --chip Sentinel-1
    python -m perf_models summary --chip all

    python -m perf_models kernel matmul --chip Discovery-1 --m 1024 --n 1024 --k 1024
    python -m perf_models kernel attention --chip Prometheus --b 1 --h 32 --s 2048 --d 128

    python -m perf_models workload llama --chip Prometheus --seq-len 2048
    python -m perf_models workload resnet50 --chip Discovery-1 --batch 8

    python -m perf_models power --chip Sentinel-1 --op nominal
    python -m perf_models power --chip all

    python -m perf_models thermal --chip Sentinel-1 --power 50

    python -m perf_models compare --chips Sentinel-1,Discovery-1 --workload llama --seq-len 512

The CLI uses argparse from the standard library only (no click/typer
dependency) so it runs in any vanilla Python 3.10+ environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Tuple

from .base import OperatingPoint, PrecisionMode
from .chips import all_chips, get_chip
from .kernels import attention, conv2d, fft, gemv, matmul
from .power import compute_power_breakdown, find_optimal_op_for_perf_per_watt, power_sweep
from .thermal import THERMAL_PROFILES, get_thermal_profile
from .workloads import (
    aes_xts_throughput, dicom_3d_cnn, kyber_768_full_round,
    llama_7b_decode, ofdm_256_tx_symbol, resnet50_forward,
)


# ============================================================================
# Helpers
# ============================================================================


def _resolve_chips(chip_arg: str):
    """Parse --chip <name|all|csv> into a list of ChipModels."""
    if chip_arg.lower() == "all":
        return all_chips()
    return [get_chip(c.strip()) for c in chip_arg.split(",")]


def _resolve_op(chip, op_arg: Optional[str]) -> OperatingPoint:
    """Pick an OperatingPoint by name or use chip's nominal."""
    if op_arg is None:
        return chip.nominal_op
    op_arg_lower = op_arg.lower()
    for op in chip.operating_points:
        if op.name.lower() == op_arg_lower:
            return op
    raise SystemExit(
        f"chip {chip.name} has no operating point named '{op_arg}'. "
        f"Available: {[op.name for op in chip.operating_points]}"
    )


def _resolve_prec(prec_arg: str) -> PrecisionMode:
    """Parse --prec <name>."""
    try:
        return PrecisionMode(prec_arg.lower())
    except ValueError:
        valid = sorted(p.value for p in PrecisionMode)
        raise SystemExit(f"unknown precision '{prec_arg}'. Valid: {valid}")


# ============================================================================
# Subcommand handlers
# ============================================================================


def cmd_summary(args) -> int:
    """Print summary table for one or more chips."""
    chips = _resolve_chips(args.chip)
    for c in chips:
        print(c.summary_table())
        print()
    return 0


def cmd_kernel(args) -> int:
    """Run a kernel-level analytical model."""
    chips = _resolve_chips(args.chip)
    prec = _resolve_prec(args.prec)
    kernel_name = args.kernel.lower()

    for c in chips:
        op = _resolve_op(c, args.op)
        if kernel_name == "matmul":
            r = matmul(c, M=args.m, N=args.n, K=args.k, prec=prec, op=op)
        elif kernel_name == "conv2d":
            r = conv2d(c, N=args.n_batch, C=args.c, H=args.h, W=args.w,
                       K=args.k, R=args.r, S=args.s, prec=prec, op=op)
        elif kernel_name == "attention":
            r = attention(c, B=args.b, H_heads=args.h_heads, S_seq=args.s_seq,
                          D_head=args.d, prec=prec, op=op)
        elif kernel_name == "fft":
            r = fft(c, N=args.n_fft, prec=prec, op=op)
        elif kernel_name == "gemv":
            r = gemv(c, M=args.m, N=args.n, prec=prec, op=op)
        else:
            print(f"unknown kernel '{args.kernel}'")
            return 2
        print(f"=== {c.name} ===")
        print(f"  {r}")
        print(f"  GFLOPS: {r.gflops:,.1f}")
        print(f"  GB/s:   {r.gb_per_s:,.1f}")
    return 0


def cmd_workload(args) -> int:
    """Run an end-to-end workload model."""
    chips = _resolve_chips(args.chip)
    prec = _resolve_prec(args.prec)

    for c in chips:
        op = _resolve_op(c, args.op)
        if args.workload == "llama":
            wr = llama_7b_decode(c, seq_len=args.seq_len, prec=prec, op=op)
        elif args.workload == "resnet50":
            wr = resnet50_forward(c, batch=args.batch, prec=prec, op=op)
        elif args.workload == "aes":
            wr = aes_xts_throughput(c, bytes_to_encrypt=args.bytes_, prec=prec, op=op)
        elif args.workload == "ofdm256":
            wr = ofdm_256_tx_symbol(c, prec=prec, op=op)
        elif args.workload == "kyber768":
            wr = kyber_768_full_round(c, prec=prec, op=op)
        elif args.workload == "dicom":
            wr = dicom_3d_cnn(c, volume_dim=args.volume, prec=prec, op=op)
        else:
            print(f"unknown workload '{args.workload}'")
            return 2
        print(wr)
        print()
    return 0


def cmd_power(args) -> int:
    """Print power breakdown."""
    chips = _resolve_chips(args.chip)
    prec = _resolve_prec(args.prec)
    for c in chips:
        if args.op:
            op = _resolve_op(c, args.op)
            bd = compute_power_breakdown(c, op, prec)
            print(bd)
        else:
            for bd in power_sweep(c, prec):
                print(bd)
                print()
    return 0


def cmd_thermal(args) -> int:
    """Project thermal behavior."""
    chips = _resolve_chips(args.chip)
    for c in chips:
        try:
            tp = get_thermal_profile(c.name)
        except KeyError:
            print(f"no thermal profile for {c.name}")
            continue
        ss_tj = tp.steady_state_t_junction(args.power)
        time_to = tp.time_to_throttle_s(args.power)
        print(f"=== {c.name} thermal projection at {args.power} W ===")
        print(f"  Profile:           {tp.name}")
        print(f"  R total:           {tp.r_total_k_per_w:.3f} K/W")
        print(f"  Tau (time const):  {tp.thermal_time_constant_s:.2f} s")
        print(f"  T_ambient:         {tp.t_ambient_c:.0f} C")
        print(f"  T_throttle:        {tp.t_throttle_c:.0f} C")
        print(f"  Steady-state Tj:   {ss_tj:.1f} C")
        print(f"  Throttle?          {'YES' if not tp.is_safe_at_steady_state(args.power) else 'no'}")
        if time_to is not None:
            print(f"  Time to throttle:  {time_to:.2f} s")
        else:
            print(f"  Time to throttle:  never (steady state safe)")
        print()
    return 0


def cmd_compare(args) -> int:
    """Compare multiple chips on one workload."""
    chips = _resolve_chips(args.chips)
    prec = _resolve_prec(args.prec)

    rows = []
    for c in chips:
        if args.workload == "llama":
            wr = llama_7b_decode(c, seq_len=args.seq_len, prec=prec)
        elif args.workload == "resnet50":
            wr = resnet50_forward(c, batch=args.batch, prec=prec)
        elif args.workload == "matmul":
            from .kernels import matmul as _matmul
            r = _matmul(c, M=args.m, N=args.n, K=args.k, prec=prec)
            from .workloads import WorkloadResult
            wr = WorkloadResult(name=f"matmul {args.m}x{args.n}x{args.k}",
                                chip_name=c.name, kernels=[r])
        else:
            print(f"unknown workload '{args.workload}'")
            return 2
        rows.append((c.name, wr.total_runtime_ns, wr.total_energy_pj, wr.avg_throughput_gflops))

    print(f"=== Comparison: {args.workload} ({prec.value}) ===")
    print(f"{'Chip':<14} {'Runtime (ns)':>15} {'Energy (pJ)':>15} {'GFLOPS':>12}")
    for name, rt, en, gflops in rows:
        print(f"{name:<14} {rt:>15,.0f} {en:>15,.0f} {gflops:>12,.1f}")
    return 0


def cmd_optimal_op(args) -> int:
    """Find best perf-per-watt operating point."""
    chips = _resolve_chips(args.chip)
    prec = _resolve_prec(args.prec)
    for c in chips:
        op = find_optimal_op_for_perf_per_watt(c, prec)
        pw = c.perf_per_watt_tops_w(op, prec)
        print(f"{c.name}: optimal op = {op.name} "
              f"({op.voltage_v:.2f} V, {op.frequency_ghz:.2f} GHz) "
              f"-> {pw:.2f} TOPS/W at {prec.value}")
    return 0


def cmd_export_json(args) -> int:
    """Export chip config as JSON for downstream tools."""
    chips = _resolve_chips(args.chip)
    payload = {c.name: c.serializable() for c in chips}
    print(json.dumps(payload, indent=2, default=str))
    return 0


# ============================================================================
# Argument parser
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zhi-perf",
        description="Zhilicon analytical performance modeling CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # summary
    p_sum = sub.add_parser("summary", help="Print chip summary tables")
    p_sum.add_argument("--chip", required=True,
                       help="Chip name, 'all', or comma-separated list")
    p_sum.set_defaults(func=cmd_summary)

    # kernel
    p_k = sub.add_parser("kernel", help="Run a kernel-level model")
    p_k.add_argument("kernel", choices=["matmul", "conv2d", "attention", "fft", "gemv"])
    p_k.add_argument("--chip", required=True)
    p_k.add_argument("--prec", default="fp16")
    p_k.add_argument("--op", default=None, help="Operating point name")
    # matmul / gemv
    p_k.add_argument("--m", type=int, default=1024)
    p_k.add_argument("--n", type=int, default=1024)
    p_k.add_argument("--k", type=int, default=1024)
    # conv2d
    p_k.add_argument("--n-batch", type=int, default=1)
    p_k.add_argument("--c", type=int, default=64)
    p_k.add_argument("--h", type=int, default=224)
    p_k.add_argument("--w", type=int, default=224)
    p_k.add_argument("--r", type=int, default=3)
    p_k.add_argument("--s", type=int, default=3)
    # attention
    p_k.add_argument("--b", type=int, default=1)
    p_k.add_argument("--h-heads", type=int, default=8)
    p_k.add_argument("--s-seq", type=int, default=512)
    p_k.add_argument("--d", type=int, default=64)
    # fft
    p_k.add_argument("--n-fft", type=int, default=256)
    p_k.set_defaults(func=cmd_kernel)

    # workload
    p_w = sub.add_parser("workload", help="Run a workload-level model")
    p_w.add_argument("workload", choices=["llama", "resnet50", "aes", "ofdm256",
                                           "kyber768", "dicom"])
    p_w.add_argument("--chip", required=True)
    p_w.add_argument("--prec", default="fp16")
    p_w.add_argument("--op", default=None)
    p_w.add_argument("--seq-len", type=int, default=2048)
    p_w.add_argument("--batch", type=int, default=1)
    p_w.add_argument("--bytes", dest="bytes_", type=int, default=1024 * 1024)
    p_w.add_argument("--volume", type=int, default=128)
    p_w.set_defaults(func=cmd_workload)

    # power
    p_p = sub.add_parser("power", help="Show power breakdown")
    p_p.add_argument("--chip", required=True)
    p_p.add_argument("--prec", default="fp16")
    p_p.add_argument("--op", default=None,
                     help="Specific op or omit for full sweep")
    p_p.set_defaults(func=cmd_power)

    # thermal
    p_t = sub.add_parser("thermal", help="Project thermal behavior")
    p_t.add_argument("--chip", required=True)
    p_t.add_argument("--power", type=float, required=True,
                     help="Sustained power in watts")
    p_t.set_defaults(func=cmd_thermal)

    # compare
    p_c = sub.add_parser("compare", help="Compare multiple chips on a workload")
    p_c.add_argument("--chips", required=True, help="Comma-separated chip names")
    p_c.add_argument("--workload", required=True,
                     choices=["llama", "resnet50", "matmul"])
    p_c.add_argument("--prec", default="fp16")
    p_c.add_argument("--seq-len", type=int, default=2048)
    p_c.add_argument("--batch", type=int, default=1)
    p_c.add_argument("--m", type=int, default=1024)
    p_c.add_argument("--n", type=int, default=1024)
    p_c.add_argument("--k", type=int, default=1024)
    p_c.set_defaults(func=cmd_compare)

    # optimal-op
    p_o = sub.add_parser("optimal-op", help="Find best perf-per-watt op")
    p_o.add_argument("--chip", required=True)
    p_o.add_argument("--prec", default="fp16")
    p_o.set_defaults(func=cmd_optimal_op)

    # export-json
    p_j = sub.add_parser("export-json", help="Export chip config as JSON")
    p_j.add_argument("--chip", required=True)
    p_j.set_defaults(func=cmd_export_json)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
