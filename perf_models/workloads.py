"""
End-to-end workload models composing per-kernel models into full traces.

Workloads:
    LLaMA-7B inference (single-token, decode phase)
    ResNet-50 forward pass (1 image, batch=1)
    AES-XTS bulk encryption (1 GB)
    Kyber-768 keygen + encap + decap
    OFDM-256 modem TX/RX symbol
    DICOM 3D-CNN inference (single 256x256x256 volume)

Each workload returns a `WorkloadResult` aggregating runtime + energy +
breakdowns across constituent kernels. Used to project full-application
perf/W and to validate against silicon measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import ChipModel, OperatingPoint, PrecisionMode
from .kernels import (
    KernelResult, attention, conv2d, fft, gemv, matmul,
)


@dataclass
class WorkloadResult:
    """Aggregated result for a multi-kernel workload."""

    name: str
    chip_name: str
    kernels: List[KernelResult] = field(default_factory=list)

    @property
    def total_flops(self) -> int:
        return sum(k.flops for k in self.kernels)

    @property
    def total_bytes(self) -> int:
        return sum(k.bytes_read + k.bytes_written for k in self.kernels)

    @property
    def total_runtime_ns(self) -> float:
        """Sequential composition: sum of per-kernel runtimes."""
        return sum(k.runtime_ns for k in self.kernels)

    @property
    def total_energy_pj(self) -> float:
        return sum(k.energy_pj for k in self.kernels)

    @property
    def avg_throughput_gflops(self) -> float:
        if self.total_runtime_ns == 0:
            return 0.0
        return self.total_flops / self.total_runtime_ns

    @property
    def avg_power_w(self) -> float:
        if self.total_runtime_ns == 0:
            return 0.0
        # energy (pJ) / time (ns) = power (mW)... = mW; convert to W
        return (self.total_energy_pj / self.total_runtime_ns) * 1e-3

    def __str__(self) -> str:
        lines = []
        lines.append(f"=== {self.name} on {self.chip_name} ===")
        lines.append(f"  Kernels:    {len(self.kernels)}")
        lines.append(f"  FLOPs:      {self.total_flops:,}")
        lines.append(f"  Bytes:      {self.total_bytes:,}")
        lines.append(f"  Runtime:    {self.total_runtime_ns:,.0f} ns "
                     f"({self.total_runtime_ns/1e6:.3f} ms)")
        lines.append(f"  Energy:     {self.total_energy_pj:,.0f} pJ "
                     f"({self.total_energy_pj*1e-9:.3f} mJ)")
        lines.append(f"  Avg power:  {self.avg_power_w:,.1f} W")
        lines.append(f"  Throughput: {self.avg_throughput_gflops:,.1f} GFLOPS")
        return "\n".join(lines)


# ============================================================================
# LLaMA-7B inference (decode phase, single token)
# ============================================================================


def llama_7b_decode(
    chip: ChipModel,
    seq_len: int = 2048,
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> WorkloadResult:
    """LLaMA-7B decode phase, single token at given context length.

    Architecture:
        32 transformer layers
        Hidden dim = 4096
        FFN intermediate = 11008
        Heads = 32, head dim = 128
        Vocab = 32000
    """
    NUM_LAYERS = 32
    HIDDEN_DIM = 4096
    FFN_DIM = 11008
    HEADS = 32
    HEAD_DIM = 128
    VOCAB = 32000

    result = WorkloadResult(name=f"LLaMA-7B decode (S={seq_len})", chip_name=chip.name)

    for _ in range(NUM_LAYERS):
        # 1. QKV projection: 3 matmuls of (1, HIDDEN) @ (HIDDEN, HIDDEN)
        result.kernels.append(matmul(chip, M=1, N=HIDDEN_DIM, K=HIDDEN_DIM, prec=prec, op=op))
        result.kernels.append(matmul(chip, M=1, N=HIDDEN_DIM, K=HIDDEN_DIM, prec=prec, op=op))
        result.kernels.append(matmul(chip, M=1, N=HIDDEN_DIM, K=HIDDEN_DIM, prec=prec, op=op))

        # 2. Attention: (B=1, H=32, S=seq_len, D=128)
        result.kernels.append(attention(chip, B=1, H_heads=HEADS, S_seq=seq_len, D_head=HEAD_DIM, prec=prec, op=op))

        # 3. Output projection: (1, HIDDEN) @ (HIDDEN, HIDDEN)
        result.kernels.append(matmul(chip, M=1, N=HIDDEN_DIM, K=HIDDEN_DIM, prec=prec, op=op))

        # 4. FFN: (1, HIDDEN) @ (HIDDEN, FFN_DIM) and (1, FFN_DIM) @ (FFN_DIM, HIDDEN)
        result.kernels.append(matmul(chip, M=1, N=FFN_DIM,    K=HIDDEN_DIM, prec=prec, op=op))
        result.kernels.append(matmul(chip, M=1, N=HIDDEN_DIM, K=FFN_DIM,    prec=prec, op=op))

    # 5. Final LM head: (1, HIDDEN) @ (HIDDEN, VOCAB)
    result.kernels.append(matmul(chip, M=1, N=VOCAB, K=HIDDEN_DIM, prec=prec, op=op))

    return result


# ============================================================================
# ResNet-50 inference (forward, batch=1)
# ============================================================================


def resnet50_forward(
    chip: ChipModel,
    batch: int = 1,
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> WorkloadResult:
    """ResNet-50 forward, batch=1.

    Simplified: we model the headline conv layers + the FC layer.
    The real network has ~50 conv layers; modeling each is overkill
    for a smoke-level performance projection. We model representative
    layers from each stage of the network.
    """
    result = WorkloadResult(name=f"ResNet-50 forward (B={batch})", chip_name=chip.name)

    # Stem: conv 7x7 stride=2, 224x224 -> 112x112
    result.kernels.append(conv2d(
        chip, N=batch, C=3, H=224, W=224, K=64, R=7, S=7,
        stride=2, padding=3, prec=prec, op=op
    ))

    # Stage 1 (conv2_x): 3 bottleneck blocks, each ~3 convs
    for _ in range(3):
        # 1x1 reduce
        result.kernels.append(conv2d(chip, N=batch, C=64,  H=56, W=56, K=64,
                                     R=1, S=1, padding=0, prec=prec, op=op))
        # 3x3
        result.kernels.append(conv2d(chip, N=batch, C=64,  H=56, W=56, K=64,
                                     R=3, S=3, padding=1, prec=prec, op=op))
        # 1x1 expand
        result.kernels.append(conv2d(chip, N=batch, C=64,  H=56, W=56, K=256,
                                     R=1, S=1, padding=0, prec=prec, op=op))

    # Stage 2 (conv3_x): 4 bottleneck blocks, 28x28
    for _ in range(4):
        result.kernels.append(conv2d(chip, N=batch, C=128, H=28, W=28, K=128,
                                     R=3, S=3, padding=1, prec=prec, op=op))

    # Stage 3 (conv4_x): 6 bottleneck blocks, 14x14
    for _ in range(6):
        result.kernels.append(conv2d(chip, N=batch, C=256, H=14, W=14, K=256,
                                     R=3, S=3, padding=1, prec=prec, op=op))

    # Stage 4 (conv5_x): 3 bottleneck blocks, 7x7
    for _ in range(3):
        result.kernels.append(conv2d(chip, N=batch, C=512, H=7, W=7, K=512,
                                     R=3, S=3, padding=1, prec=prec, op=op))

    # FC: 2048 -> 1000
    result.kernels.append(matmul(chip, M=batch, N=1000, K=2048, prec=prec, op=op))

    return result


# ============================================================================
# AES-XTS bulk encryption
# ============================================================================


def aes_xts_throughput(
    chip: ChipModel,
    bytes_to_encrypt: int = 1024 * 1024 * 1024,  # 1 GB default
    prec: PrecisionMode = PrecisionMode.INT8,
    op: Optional[OperatingPoint] = None,
) -> WorkloadResult:
    """AES-256-XTS throughput model.

    Simplified: each AES round = 16 byte block, ~14 rounds = ~16*14 = 224
    "MACs" per block (each MAC = byte-wide ops in S-box + ShiftRows +
    MixColumns). In a real engine these are bit-level ops, not MACs;
    we approximate to keep the framework consistent.
    """
    if op is None:
        op = chip.nominal_op

    BLOCK_SIZE = 16  # AES block in bytes
    num_blocks = bytes_to_encrypt // BLOCK_SIZE

    # Approximate AES round count
    ROUNDS = 14   # AES-256
    OPS_PER_ROUND_PER_BLOCK = 16  # 16 bytes processed per round

    flops = num_blocks * ROUNDS * OPS_PER_ROUND_PER_BLOCK
    bytes_read = bytes_to_encrypt
    bytes_written = bytes_to_encrypt

    # Use matmul as a vehicle for the analytical roofline (close enough
    # for an approximation; real model would use the AES-engine model).
    pseudo_M = max(1, num_blocks // 256)
    pseudo_K = 256
    pseudo_N = 256
    k = matmul(chip, M=pseudo_M, N=pseudo_N, K=pseudo_K, prec=prec, op=op)
    # Override the energy/flops with AES-specific values
    k = KernelResult(
        name=f"aes_xts_{bytes_to_encrypt}_bytes",
        flops=flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        runtime_ns=k.runtime_ns,
        energy_pj=k.energy_pj,
        bound=k.bound,
    )
    result = WorkloadResult(
        name=f"AES-XTS {bytes_to_encrypt:,} bytes",
        chip_name=chip.name,
        kernels=[k],
    )
    return result


# ============================================================================
# OFDM-256 modem (TX path: IFFT + CP insertion + DAC)
# ============================================================================


def ofdm_256_tx_symbol(
    chip: ChipModel,
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> WorkloadResult:
    """One OFDM-256 TX symbol: 256-point IFFT + CP insertion."""
    result = WorkloadResult(name="OFDM-256 TX symbol", chip_name=chip.name)
    # The 256-point IFFT is the dominant compute
    result.kernels.append(fft(chip, N=256, prec=prec, op=op))
    return result


# ============================================================================
# Kyber-768 PQC: keygen + encap + decap
# ============================================================================


def kyber_768_full_round(
    chip: ChipModel,
    prec: PrecisionMode = PrecisionMode.INT16,
    op: Optional[OperatingPoint] = None,
) -> WorkloadResult:
    """Full Kyber-768 round: keygen + encap + decap.

    Each operation involves NTT-based polynomial multiplications.
    Polynomial degree N=256, q=3329, k=3 (Kyber-768).
    """
    if op is None:
        op = chip.nominal_op

    result = WorkloadResult(name="Kyber-768 keygen+encap+decap", chip_name=chip.name)

    # Approximate: each Kyber operation does ~k^2 NTTs
    KYBER_N = 256
    K_PARAM = 3      # Kyber-768

    # keygen: ~k * NTT, ~k * MUL
    # encap:  ~k^2 * NTT + ~k * MUL
    # decap:  ~k * NTT + ~k^2 * MUL
    total_ntts = K_PARAM + K_PARAM * K_PARAM + K_PARAM
    for _ in range(total_ntts):
        result.kernels.append(fft(chip, N=KYBER_N, prec=prec, op=op))

    return result


# ============================================================================
# DICOM 3D CNN inference (single volume)
# ============================================================================


def dicom_3d_cnn(
    chip: ChipModel,
    volume_dim: int = 256,    # 256x256x256 voxel volume
    prec: PrecisionMode = PrecisionMode.FP16,
    op: Optional[OperatingPoint] = None,
) -> WorkloadResult:
    """3D-CNN inference for a single medical volume.

    Modeled as 5 cascaded 3D convs (no batch dim handled in this 2D
    framework; we approximate as a 2D conv on a volume-sized input).
    """
    result = WorkloadResult(name=f"DICOM 3D-CNN ({volume_dim}^3)", chip_name=chip.name)

    # 5 cascaded convs, decreasing spatial size
    sizes = [
        (1,  16, volume_dim,    volume_dim,    32),
        (1,  32, volume_dim//2, volume_dim//2, 64),
        (1,  64, volume_dim//4, volume_dim//4, 128),
        (1, 128, volume_dim//8, volume_dim//8, 256),
        (1, 256, volume_dim//16, volume_dim//16, 512),
    ]
    for N, C, H, W, K in sizes:
        result.kernels.append(conv2d(
            chip, N=N, C=C, H=H, W=W, K=K, R=3, S=3,
            stride=1, padding=1, prec=prec, op=op
        ))
    # Final classifier
    result.kernels.append(matmul(chip, M=1, N=10, K=512, prec=prec, op=op))
    return result
