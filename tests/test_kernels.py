"""Tests for kernel-level analytical models."""

from __future__ import annotations

import math
import pytest

from perf_models.base import PrecisionMode
from perf_models.chips import (
    Sentinel1Model,
    Discovery1Model,
    PrometheusModel,
)
from perf_models.kernels import (
    KernelResult,
    Bound,
    matmul,
    conv2d,
    attention,
    fft,
    gemv,
)


# ============================================================================
# matmul
# ============================================================================


class TestMatmul:
    def test_simple_matmul(self):
        chip = Discovery1Model()
        r = matmul(chip, M=128, N=128, K=128, prec=PrecisionMode.FP16)
        assert r.flops == 2 * 128 * 128 * 128
        assert r.bytes_read == (128 * 128 + 128 * 128) * 2  # FP16 = 2 bytes
        assert r.bytes_written == 128 * 128 * 2
        assert r.runtime_ns > 0
        assert r.energy_pj > 0

    def test_matmul_int8_smaller_bytes(self):
        chip = Discovery1Model()
        r_fp16 = matmul(chip, 128, 128, 128, prec=PrecisionMode.FP16)
        r_int8 = matmul(chip, 128, 128, 128, prec=PrecisionMode.INT8)
        # INT8 = 1 byte, FP16 = 2 bytes
        assert r_int8.bytes_read == r_fp16.bytes_read // 2
        assert r_int8.bytes_written == r_fp16.bytes_written // 2

    def test_matmul_negative_shape_rejected(self):
        chip = Discovery1Model()
        with pytest.raises(ValueError):
            matmul(chip, -1, 128, 128)
        with pytest.raises(ValueError):
            matmul(chip, 128, 0, 128)
        with pytest.raises(ValueError):
            matmul(chip, 128, 128, -5)

    def test_large_matmul_compute_bound(self):
        """Large square matmul has high arithmetic intensity → compute-bound."""
        chip = Discovery1Model()
        r = matmul(chip, M=4096, N=4096, K=4096, prec=PrecisionMode.FP16)
        # AI = 2*M*N*K / (M*K + K*N + M*N)*2  ≈ K when M=N=K large
        assert r.bound == Bound.COMPUTE

    def test_matmul_flops_scale_with_dimensions(self):
        chip = Sentinel1Model()
        r1 = matmul(chip, 64, 64, 64, prec=PrecisionMode.FP16)
        r2 = matmul(chip, 128, 128, 128, prec=PrecisionMode.FP16)
        # 8x more flops (2x in each of 3 dims)
        assert r2.flops == 8 * r1.flops

    def test_arithmetic_intensity(self):
        chip = PrometheusModel()
        r = matmul(chip, 1024, 1024, 1024, prec=PrecisionMode.FP16)
        # AI for square matmul ≈ M when M=N=K (dimensional analysis)
        ai = r.arithmetic_intensity
        # Loose check: AI should be in the hundreds for 1024-cube
        assert 100 <= ai <= 1000

    def test_runtime_at_higher_freq_is_shorter(self):
        chip = Discovery1Model()
        low_op = chip.lowest_power_op
        high_op = chip.highest_perf_op
        r_low = matmul(chip, 1024, 1024, 1024, prec=PrecisionMode.FP16, op=low_op)
        r_high = matmul(chip, 1024, 1024, 1024, prec=PrecisionMode.FP16, op=high_op)
        assert r_high.runtime_ns < r_low.runtime_ns


# ============================================================================
# conv2d
# ============================================================================


class TestConv2d:
    def test_simple_conv(self):
        chip = Discovery1Model()
        r = conv2d(chip, N=1, C=3, H=224, W=224, K=64, R=3, S=3,
                   stride=1, padding=1, prec=PrecisionMode.FP16)
        # H' = (224 + 2 - 3) / 1 + 1 = 224
        # flops = 2 * 1 * 64 * 224 * 224 * 3 * 3 * 3 = 173,408,256
        expected_flops = 2 * 64 * 224 * 224 * 3 * 3 * 3
        assert r.flops == expected_flops
        assert r.runtime_ns > 0

    def test_conv_negative_output_raises(self):
        """Padding too small for kernel size → negative output dim."""
        chip = Discovery1Model()
        with pytest.raises(ValueError):
            conv2d(chip, N=1, C=3, H=4, W=4, K=8, R=7, S=7, padding=0)

    def test_resnet50_first_layer_size(self):
        """ResNet-50 first layer: 224x224 → 112x112, K=64, 7x7."""
        chip = Discovery1Model()
        r = conv2d(chip, N=1, C=3, H=224, W=224, K=64,
                   R=7, S=7, stride=2, padding=3, prec=PrecisionMode.FP16)
        assert r.flops > 0
        # Output: (224 + 6 - 7)/2 + 1 = 112
        # flops = 2*1*64*112*112*3*7*7 = 236MFLOPS
        assert 200_000_000 < r.flops < 300_000_000

    def test_conv_int8_smaller_bytes(self):
        chip = Discovery1Model()
        r_fp16 = conv2d(chip, 1, 3, 224, 224, 64, prec=PrecisionMode.FP16)
        r_int8 = conv2d(chip, 1, 3, 224, 224, 64, prec=PrecisionMode.INT8)
        assert r_int8.bytes_read < r_fp16.bytes_read


# ============================================================================
# attention
# ============================================================================


class TestAttention:
    def test_simple_attention(self):
        chip = PrometheusModel()
        r = attention(chip, B=1, H_heads=8, S_seq=128, D_head=64,
                      prec=PrecisionMode.FP16)
        # flops = 2*(QK^T) + 2*(softmax->V) + softmax overhead
        #       = 2*1*8*128*128*64 + 2*1*8*128*128*64 + 3*1*8*128*128
        expected = 2 * (2 * 1 * 8 * 128 * 128 * 64) + 3 * 1 * 8 * 128 * 128
        assert r.flops == expected

    def test_attention_quadratic_in_seq_length(self):
        """Attention is O(S^2 * D); doubling S = 4x flops (approx)."""
        chip = PrometheusModel()
        r1 = attention(chip, B=1, H_heads=8, S_seq=128, D_head=64)
        r2 = attention(chip, B=1, H_heads=8, S_seq=256, D_head=64)
        # 4x not exact due to softmax term, but ratio should be close
        ratio = r2.flops / r1.flops
        assert 3.5 < ratio < 4.5

    def test_attention_runtime_positive(self):
        chip = PrometheusModel()
        r = attention(chip, B=1, H_heads=8, S_seq=2048, D_head=128,
                      prec=PrecisionMode.FP16)
        assert r.runtime_ns > 0


# ============================================================================
# fft
# ============================================================================


class TestFFT:
    def test_simple_fft(self):
        chip = Discovery1Model()
        r = fft(chip, N=1024, prec=PrecisionMode.FP16)
        # flops = 5 * 1024 * 10 = 51200
        assert r.flops == 5 * 1024 * 10

    def test_non_power_of_2_raises(self):
        chip = Discovery1Model()
        with pytest.raises(ValueError):
            fft(chip, N=300)

    def test_zero_n_raises(self):
        chip = Discovery1Model()
        with pytest.raises(ValueError):
            fft(chip, N=0)

    def test_fft_logarithmic_scaling(self):
        """FFT should scale O(N log N), so doubling N = ~2.x flops increase."""
        chip = Discovery1Model()
        r1 = fft(chip, N=1024)
        r2 = fft(chip, N=2048)
        ratio = r2.flops / r1.flops
        # 2 * 11 / 1 * 10 = 2.2
        assert 2.0 < ratio < 2.5


# ============================================================================
# gemv
# ============================================================================


class TestGemv:
    def test_simple_gemv(self):
        chip = Sentinel1Model()
        r = gemv(chip, M=512, N=512, prec=PrecisionMode.FP16)
        assert r.flops == 2 * 512 * 512

    def test_gemv_likely_memory_bound(self):
        """GEMV is famously memory-bound at large M, N."""
        chip = Sentinel1Model()
        r = gemv(chip, M=8192, N=8192, prec=PrecisionMode.FP16)
        # Arithmetic intensity = 2*M*N / ((M*N + N + M)*2) ≈ 1
        # Should be memory-bound on most chips
        assert r.bound == Bound.MEMORY

    def test_gemv_negative_shape_raises(self):
        chip = Sentinel1Model()
        with pytest.raises(ValueError):
            gemv(chip, M=0, N=128)
        with pytest.raises(ValueError):
            gemv(chip, M=128, N=-1)


# ============================================================================
# KernelResult
# ============================================================================


class TestKernelResult:
    def test_arithmetic_intensity(self):
        r = KernelResult(
            name="test", flops=1000, bytes_read=10, bytes_written=10,
            runtime_ns=100.0, energy_pj=200.0, bound=Bound.COMPUTE,
        )
        # AI = 1000 / (10 + 10) = 50
        assert r.arithmetic_intensity == 50.0

    def test_arithmetic_intensity_zero_bytes(self):
        r = KernelResult(
            name="test", flops=1000, bytes_read=0, bytes_written=0,
            runtime_ns=10.0, energy_pj=10.0, bound=Bound.COMPUTE,
        )
        assert r.arithmetic_intensity == float("inf")

    def test_gflops(self):
        r = KernelResult(
            name="test", flops=10_000, bytes_read=1, bytes_written=1,
            runtime_ns=10.0, energy_pj=1.0, bound=Bound.COMPUTE,
        )
        # 10000 flops / 10 ns = 1000 flops/ns = 1 GFLOPS
        assert r.gflops == 1000.0

    def test_gb_per_s(self):
        r = KernelResult(
            name="test", flops=1, bytes_read=500, bytes_written=500,
            runtime_ns=1.0, energy_pj=1.0, bound=Bound.MEMORY,
        )
        # 1000 bytes in 1 ns = 1000 bytes/ns = 1000 GB/s
        assert r.gb_per_s == 1000.0

    def test_str(self):
        r = KernelResult(
            name="my_kernel", flops=100, bytes_read=10, bytes_written=10,
            runtime_ns=1.0, energy_pj=2.0, bound=Bound.MEMORY,
        )
        s = str(r)
        assert "my_kernel" in s
        assert "100" in s
        assert "memory" in s


# ============================================================================
# Cross-chip kernel comparison (energy and runtime sanity checks)
# ============================================================================


class TestCrossChipKernels:
    def test_prometheus_faster_than_sentinel(self):
        """Big matmul should run faster on the larger Prometheus chip."""
        s = Sentinel1Model()
        p = PrometheusModel()
        m_size = 1024
        r_s = matmul(s, m_size, m_size, m_size, prec=PrecisionMode.FP16)
        r_p = matmul(p, m_size, m_size, m_size, prec=PrecisionMode.FP16)
        assert r_p.runtime_ns < r_s.runtime_ns

    def test_int8_faster_or_equal_to_fp16(self):
        """At equal MAC count, INT8 isn't slower than FP16 (model assumes
        same MAC throughput; the win is in energy)."""
        chip = Discovery1Model()
        size = 512
        r_fp16 = matmul(chip, size, size, size, prec=PrecisionMode.FP16)
        r_int8 = matmul(chip, size, size, size, prec=PrecisionMode.INT8)
        # INT8 should at least not be slower (memory-bound regime, INT8 wins)
        assert r_int8.runtime_ns <= r_fp16.runtime_ns + 1e-6

    def test_int8_more_energy_efficient_than_fp16(self):
        """Same matmul should consume less energy at INT8 than FP16."""
        chip = Discovery1Model()
        size = 512
        r_fp16 = matmul(chip, size, size, size, prec=PrecisionMode.FP16)
        r_int8 = matmul(chip, size, size, size, prec=PrecisionMode.INT8)
        assert r_int8.energy_pj < r_fp16.energy_pj
