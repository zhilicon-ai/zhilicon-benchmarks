"""
Per-kernel analytical performance models.

Each kernel function takes a `ChipModel` plus the kernel's shape parameters
and returns a `KernelResult` with:
* `flops`         : total floating-point operations
* `bytes_read`    : total bytes read from memory
* `bytes_written` : total bytes written to memory
* `runtime_ns`    : projected runtime in nanoseconds
* `energy_pj`     : projected energy consumption
* `arithmetic_intensity`: flops / (bytes_read + bytes_written)
* `bound`         : "compute" or "memory" (whichever is the binding constraint)

The roofline model is the standard analysis tool here: a kernel is
compute-bound if its arithmetic intensity exceeds the chip's flops/byte
ratio at peak; otherwise it is memory-bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .base import (
    ChipModel,
    OperatingPoint,
    PrecisionMode,
    ENERGY_PER_MAC_PJ,
)


class Bound(Enum):
    COMPUTE = "compute"
    MEMORY  = "memory"
    LATENCY = "latency"


@dataclass(frozen=True)
class KernelResult:
    """Result of analytical kernel modeling."""
    name: str
    flops: int
    bytes_read: int
    bytes_written: int
    runtime_ns: float
    energy_pj: float
    bound: Bound

    @property
    def arithmetic_intensity(self) -> float:
        """flops per byte transferred."""
        total_bytes = self.bytes_read + self.bytes_written
        return self.flops / total_bytes if total_bytes > 0 else float("inf")

    @property
    def gflops(self) -> float:
        """Achieved GFLOPS = flops / runtime."""
        return (self.flops / self.runtime_ns) if self.runtime_ns > 0 else 0.0

    @property
    def gb_per_s(self) -> float:
        """Achieved memory bandwidth in GB/s."""
        total_bytes = self.bytes_read + self.bytes_written
        return total_bytes / self.runtime_ns if self.runtime_ns > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.flops:,} flops, "
            f"{self.bytes_read+self.bytes_written:,} bytes, "
            f"{self.runtime_ns:.1f} ns, "
            f"{self.energy_pj:.1f} pJ, "
            f"AI={self.arithmetic_intensity:.2f}, "
            f"bound={self.bound.value}"
        )


def _bytes_per_element(prec: PrecisionMode) -> int:
    """Bytes per element for a given precision."""
    table = {
        PrecisionMode.FP4_E2M1: 1,         # Packed 2 elements per byte; round up
        PrecisionMode.FP8_E4M3: 1,
        PrecisionMode.FP8_E5M2: 1,
        PrecisionMode.BF16:     2,
        PrecisionMode.FP16:     2,
        PrecisionMode.TF32:     4,
        PrecisionMode.FP32:     4,
        PrecisionMode.FP64:     8,
        PrecisionMode.INT4:     1,
        PrecisionMode.INT8:     1,
        PrecisionMode.INT16:    2,
        PrecisionMode.INT32:    4,
    }
    return table[prec]


def _project_runtime(
    chip: ChipModel,
    op: OperatingPoint,
    flops: int,
    bytes_total: int,
    prec: PrecisionMode,
) -> tuple[float, Bound]:
    """Roofline-style runtime projection.

    Returns (runtime_ns, bound). The runtime is the max of compute time
    and memory time -- whichever is the binding constraint.
    """
    # Compute time
    peak_tops_at_op = (
        chip.fabric.peak_throughput_tops(prec)
        * (op.frequency_ghz / chip.fabric.peak_freq_ghz)
    )
    achievable_tops = peak_tops_at_op * op.activity_factor
    if achievable_tops <= 0:
        compute_ns = float("inf")
    else:
        # tops = 1e12 ops/s -> ops/ns = tops * 1e3 = tops * 1000
        compute_ns = flops / (achievable_tops * 1000.0)

    # Memory time (assume served from HBM as worst case)
    if chip.memory.hbm_bw_gb_s <= 0:
        memory_ns = float("inf")
    else:
        # GB/s = bytes/ns (1 GB/s = 1 byte/ns)
        memory_ns = bytes_total / chip.memory.hbm_bw_gb_s

    if compute_ns >= memory_ns:
        return compute_ns, Bound.COMPUTE
    return memory_ns, Bound.MEMORY


def _project_energy(
    chip: ChipModel,
    op: OperatingPoint,
    flops: int,
    bytes_read: int,
    bytes_written: int,
    prec: PrecisionMode,
) -> float:
    """Energy in picojoules.

    Energy = compute_energy + memory_energy + interconnect_energy.

    Compute energy = flops * energy_per_op (precision-specific, voltage-scaled).
    Memory energy = bytes_read * read_energy_per_byte + bytes_written * write_energy_per_byte.
    """
    # Each "flop" actually decomposes into a multiply + add; treat flops as
    # 2 ops per MAC, so (flops / 2) MAC events.
    macs = flops // 2
    # Voltage scaling: Pdyn ~ V^2, so energy/op ~ V^2
    v_scale_sq = (op.voltage_mv / chip.process.nominal_voltage_mv) ** 2.0
    energy_per_mac = chip.fabric.get_energy_pj(prec) * v_scale_sq

    compute_energy_pj = macs * energy_per_mac

    # Memory energy: assume data comes from HBM, scaled by hierarchy hits
    memory_energy_pj = (
        (bytes_read / 64.0) * chip.memory.hbm_read_energy_pj
        + (bytes_written / 64.0) * chip.memory.hbm_write_energy_pj
    )
    return compute_energy_pj + memory_energy_pj


# ============================================================================
# Specific kernel models
# ============================================================================


def matmul(
    chip: ChipModel,
    M: int, N: int, K: int,
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> KernelResult:
    """C[M,N] = A[M,K] @ B[K,N].

    flops = 2 * M * N * K (one mul + one add per output element).
    bytes_read = (M*K + K*N) * sizeof(prec)
    bytes_written = M*N * sizeof(prec)
    """
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"matmul shapes must be positive: {M}x{N}x{K}")

    if op is None:
        op = chip.nominal_op

    bpe = _bytes_per_element(prec)
    flops = 2 * M * N * K
    bytes_read = (M * K + K * N) * bpe
    bytes_written = M * N * bpe
    bytes_total = bytes_read + bytes_written

    runtime_ns, bound = _project_runtime(chip, op, flops, bytes_total, prec)
    energy_pj = _project_energy(chip, op, flops, bytes_read, bytes_written, prec)

    return KernelResult(
        name=f"matmul_{M}x{N}x{K}_{prec.value}",
        flops=flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        runtime_ns=runtime_ns,
        energy_pj=energy_pj,
        bound=bound,
    )


def conv2d(
    chip: ChipModel,
    N: int, C: int, H: int, W: int, K: int,
    R: int = 3, S: int = 3,
    stride: int = 1, padding: int = 1,
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> KernelResult:
    """Convolution: input N x C x H x W, kernel K x C x R x S.

    Output:
      H' = (H + 2*padding - R) // stride + 1
      W' = (W + 2*padding - S) // stride + 1
    flops = 2 * N * K * H' * W' * C * R * S
    """
    if op is None:
        op = chip.nominal_op

    Hp = (H + 2 * padding - R) // stride + 1
    Wp = (W + 2 * padding - S) // stride + 1
    if Hp <= 0 or Wp <= 0:
        raise ValueError("conv2d: output shape negative; check padding/stride")

    bpe = _bytes_per_element(prec)
    flops = 2 * N * K * Hp * Wp * C * R * S
    # Input + weights + output
    bytes_read = (N * C * H * W + K * C * R * S) * bpe
    bytes_written = N * K * Hp * Wp * bpe
    bytes_total = bytes_read + bytes_written

    runtime_ns, bound = _project_runtime(chip, op, flops, bytes_total, prec)
    energy_pj = _project_energy(chip, op, flops, bytes_read, bytes_written, prec)

    return KernelResult(
        name=f"conv2d_N{N}_C{C}_H{H}_W{W}_K{K}_R{R}S{S}_{prec.value}",
        flops=flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        runtime_ns=runtime_ns,
        energy_pj=energy_pj,
        bound=bound,
    )


def attention(
    chip: ChipModel,
    B: int, H_heads: int, S_seq: int, D_head: int,
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> KernelResult:
    """Multi-head self-attention.

    Q, K, V each [B, H_heads, S_seq, D_head].
    Score = Q @ K^T / sqrt(D_head)   -> [B, H_heads, S_seq, S_seq]
    Probs = softmax(Score)
    Output = Probs @ V               -> [B, H_heads, S_seq, D_head]

    flops ~ 4 * B * H * S^2 * D + (softmax has ~3 * B * H * S^2 ops)
    """
    if op is None:
        op = chip.nominal_op

    bpe = _bytes_per_element(prec)
    matmul_qkt_flops = 2 * B * H_heads * S_seq * S_seq * D_head
    matmul_av_flops  = 2 * B * H_heads * S_seq * S_seq * D_head
    softmax_flops    = 3 * B * H_heads * S_seq * S_seq  # exp, sum, divide
    flops = matmul_qkt_flops + matmul_av_flops + softmax_flops

    qkv_bytes = 3 * B * H_heads * S_seq * D_head * bpe   # Q, K, V
    attn_bytes = B * H_heads * S_seq * S_seq * bpe        # Score / Probs
    out_bytes  = B * H_heads * S_seq * D_head * bpe

    bytes_read = qkv_bytes
    bytes_written = attn_bytes + out_bytes
    bytes_total = bytes_read + bytes_written

    runtime_ns, bound = _project_runtime(chip, op, flops, bytes_total, prec)
    energy_pj = _project_energy(chip, op, flops, bytes_read, bytes_written, prec)

    return KernelResult(
        name=f"attention_B{B}_H{H_heads}_S{S_seq}_D{D_head}_{prec.value}",
        flops=flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        runtime_ns=runtime_ns,
        energy_pj=energy_pj,
        bound=bound,
    )


def fft(
    chip: ChipModel,
    N: int,
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> KernelResult:
    """N-point complex FFT (Cooley-Tukey radix-2).

    flops ~= 5 * N * log2(N)  (3 mul + 2 add per butterfly, complex
                                arithmetic = 4 mul + 2 add real).
    bytes = 2 * N * sizeof(prec)  (in-place, complex = 2 reals).
    """
    import math

    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(f"fft N must be positive power of 2, got {N}")

    if op is None:
        op = chip.nominal_op

    bpe = _bytes_per_element(prec)
    flops = 5 * N * int(math.log2(N))
    bytes_read = 2 * N * bpe                       # complex = 2 reals
    bytes_written = 2 * N * bpe
    bytes_total = bytes_read + bytes_written

    runtime_ns, bound = _project_runtime(chip, op, flops, bytes_total, prec)
    energy_pj = _project_energy(chip, op, flops, bytes_read, bytes_written, prec)

    return KernelResult(
        name=f"fft_{N}_{prec.value}",
        flops=flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        runtime_ns=runtime_ns,
        energy_pj=energy_pj,
        bound=bound,
    )


def gemv(
    chip: ChipModel,
    M: int, N: int,
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> KernelResult:
    """y[M] = A[M,N] @ x[N].

    flops = 2 * M * N
    bytes = (M*N + N) * sizeof(prec)
    GEMV is famously memory-bound at any reasonable size.
    """
    if M <= 0 or N <= 0:
        raise ValueError(f"gemv shapes must be positive: {M}x{N}")
    if op is None:
        op = chip.nominal_op

    bpe = _bytes_per_element(prec)
    flops = 2 * M * N
    bytes_read = (M * N + N) * bpe
    bytes_written = M * bpe
    bytes_total = bytes_read + bytes_written

    runtime_ns, bound = _project_runtime(chip, op, flops, bytes_total, prec)
    energy_pj = _project_energy(chip, op, flops, bytes_read, bytes_written, prec)

    return KernelResult(
        name=f"gemv_{M}x{N}_{prec.value}",
        flops=flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        runtime_ns=runtime_ns,
        energy_pj=energy_pj,
        bound=bound,
    )
