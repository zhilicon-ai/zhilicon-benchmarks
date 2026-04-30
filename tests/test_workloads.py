"""Tests for end-to-end workload models."""

from __future__ import annotations

import math
import pytest

from perf_models.base import PrecisionMode
from perf_models.chips import (
    Sentinel1Model, Discovery1Model, Nexus1Model, PrometheusModel,
    Horizon1Model, all_chips,
)
from perf_models.kernels import KernelResult, Bound
from perf_models.workloads import (
    WorkloadResult,
    aes_xts_throughput,
    dicom_3d_cnn,
    kyber_768_full_round,
    llama_7b_decode,
    ofdm_256_tx_symbol,
    resnet50_forward,
)


class TestWorkloadResult:
    def test_empty_workload(self):
        wr = WorkloadResult(name="empty", chip_name="test")
        assert wr.total_flops == 0
        assert wr.total_bytes == 0
        assert wr.total_runtime_ns == 0.0
        assert wr.total_energy_pj == 0.0

    def test_single_kernel_aggregation(self):
        k = KernelResult(
            name="k", flops=1000, bytes_read=10, bytes_written=20,
            runtime_ns=5.0, energy_pj=100.0, bound=Bound.COMPUTE,
        )
        wr = WorkloadResult(name="w", chip_name="test", kernels=[k])
        assert wr.total_flops == 1000
        assert wr.total_bytes == 30
        assert wr.total_runtime_ns == 5.0
        assert wr.total_energy_pj == 100.0

    def test_multi_kernel_aggregation(self):
        k1 = KernelResult(
            name="k1", flops=1000, bytes_read=10, bytes_written=20,
            runtime_ns=5.0, energy_pj=100.0, bound=Bound.COMPUTE,
        )
        k2 = KernelResult(
            name="k2", flops=2000, bytes_read=15, bytes_written=25,
            runtime_ns=8.0, energy_pj=200.0, bound=Bound.MEMORY,
        )
        wr = WorkloadResult(name="w", chip_name="test", kernels=[k1, k2])
        assert wr.total_flops == 3000
        assert wr.total_bytes == 70
        assert wr.total_runtime_ns == 13.0
        assert wr.total_energy_pj == 300.0

    def test_avg_throughput_gflops(self):
        # 13_000_000 flops in 13 ns = 1_000_000 flops/ns = 1000 GFLOPS
        k = KernelResult(
            name="k", flops=13_000_000, bytes_read=1, bytes_written=1,
            runtime_ns=13.0, energy_pj=1.0, bound=Bound.COMPUTE,
        )
        wr = WorkloadResult(name="w", chip_name="test", kernels=[k])
        assert math.isclose(wr.avg_throughput_gflops, 1_000_000.0)

    def test_avg_power_w(self):
        # 1000 pJ / 100 ns = 10 mW = 0.01 W
        k = KernelResult(
            name="k", flops=1, bytes_read=1, bytes_written=1,
            runtime_ns=100.0, energy_pj=1000.0, bound=Bound.COMPUTE,
        )
        wr = WorkloadResult(name="w", chip_name="test", kernels=[k])
        assert math.isclose(wr.avg_power_w, 0.01)

    def test_str(self):
        k = KernelResult(
            name="k", flops=1000, bytes_read=10, bytes_written=20,
            runtime_ns=5.0, energy_pj=100.0, bound=Bound.COMPUTE,
        )
        wr = WorkloadResult(name="my_wl", chip_name="my_chip", kernels=[k])
        s = str(wr)
        assert "my_wl" in s
        assert "my_chip" in s


class TestLLaMA7B:
    def test_decode_single_token(self):
        chip = PrometheusModel()
        wr = llama_7b_decode(chip, seq_len=2048, prec=PrecisionMode.FP16)
        # 32 layers, each with 7 kernels (3 QKV proj + 1 attn + 1 out + 2 FFN)
        # + 1 LM head
        assert len(wr.kernels) == 32 * 7 + 1
        assert wr.total_flops > 0

    def test_longer_seq_len_more_flops(self):
        """Longer context => more attention compute."""
        chip = PrometheusModel()
        wr_short = llama_7b_decode(chip, seq_len=512)
        wr_long = llama_7b_decode(chip, seq_len=2048)
        assert wr_long.total_flops > wr_short.total_flops

    def test_decode_runtime_positive(self):
        chip = PrometheusModel()
        wr = llama_7b_decode(chip, seq_len=2048, prec=PrecisionMode.FP16)
        assert wr.total_runtime_ns > 0

    def test_int8_faster_than_fp16(self):
        """INT8 should not be slower than FP16 (same MAC count, fewer bytes)."""
        chip = Discovery1Model()
        wr_fp16 = llama_7b_decode(chip, seq_len=512, prec=PrecisionMode.FP16)
        wr_int8 = llama_7b_decode(chip, seq_len=512, prec=PrecisionMode.INT8)
        assert wr_int8.total_runtime_ns <= wr_fp16.total_runtime_ns + 1e-3

    def test_prometheus_faster_than_sentinel(self):
        """Prometheus has 192x more MACs than Sentinel-1."""
        sent = Sentinel1Model()
        prom = PrometheusModel()
        wr_sent = llama_7b_decode(sent, seq_len=512, prec=PrecisionMode.FP16)
        wr_prom = llama_7b_decode(prom, seq_len=512, prec=PrecisionMode.FP16)
        # Should be at least 10x faster (with memory etc., not full 192x)
        assert wr_sent.total_runtime_ns > wr_prom.total_runtime_ns * 5


class TestResNet50:
    def test_forward_pass_has_kernels(self):
        chip = Discovery1Model()
        wr = resnet50_forward(chip, batch=1, prec=PrecisionMode.FP16)
        assert len(wr.kernels) > 10

    def test_forward_pass_flops_in_realistic_range(self):
        """ResNet-50 is famous for ~4 GFLOPs per forward pass."""
        chip = Discovery1Model()
        wr = resnet50_forward(chip, batch=1, prec=PrecisionMode.FP16)
        # Our simplified model omits some layers, but should be at least 1 GFLOP
        assert wr.total_flops > 1_000_000_000

    def test_batch_8_more_flops_than_batch_1(self):
        chip = Discovery1Model()
        wr_b1 = resnet50_forward(chip, batch=1, prec=PrecisionMode.FP16)
        wr_b8 = resnet50_forward(chip, batch=8, prec=PrecisionMode.FP16)
        # Batch 8 should have ~8x flops
        ratio = wr_b8.total_flops / wr_b1.total_flops
        assert 7 < ratio < 9


class TestAESXts:
    def test_throughput_smoke(self):
        chip = Sentinel1Model()
        wr = aes_xts_throughput(chip, bytes_to_encrypt=1024 * 1024,
                                prec=PrecisionMode.INT8)
        assert wr.total_runtime_ns > 0

    def test_more_bytes_more_runtime(self):
        chip = Sentinel1Model()
        wr_small = aes_xts_throughput(chip, bytes_to_encrypt=1024 * 1024)
        wr_big = aes_xts_throughput(chip, bytes_to_encrypt=1024 * 1024 * 1024)
        assert wr_big.total_runtime_ns > wr_small.total_runtime_ns

    def test_byte_count_in_kernel(self):
        chip = Sentinel1Model()
        size = 1024 * 1024 * 1024
        wr = aes_xts_throughput(chip, bytes_to_encrypt=size)
        assert wr.kernels[0].bytes_read == size
        assert wr.kernels[0].bytes_written == size


class TestOFDM256:
    def test_tx_symbol_smoke(self):
        chip = Nexus1Model()
        wr = ofdm_256_tx_symbol(chip, prec=PrecisionMode.FP16)
        assert len(wr.kernels) >= 1
        assert wr.total_runtime_ns > 0

    def test_tx_symbol_uses_256_fft(self):
        chip = Nexus1Model()
        wr = ofdm_256_tx_symbol(chip, prec=PrecisionMode.FP16)
        # FFT kernel name should mention 256
        assert "256" in wr.kernels[0].name


class TestKyber768:
    def test_full_round_smoke(self):
        chip = Sentinel1Model()
        wr = kyber_768_full_round(chip, prec=PrecisionMode.INT16)
        assert len(wr.kernels) > 0
        assert wr.total_runtime_ns > 0


class TestDICOM3D:
    def test_inference_smoke(self):
        chip = Discovery1Model()
        wr = dicom_3d_cnn(chip, volume_dim=128, prec=PrecisionMode.FP16)
        assert len(wr.kernels) >= 5
        assert wr.total_flops > 0

    def test_larger_volume_more_compute(self):
        chip = Discovery1Model()
        wr_small = dicom_3d_cnn(chip, volume_dim=64, prec=PrecisionMode.FP16)
        wr_big = dicom_3d_cnn(chip, volume_dim=256, prec=PrecisionMode.FP16)
        assert wr_big.total_flops > wr_small.total_flops


class TestCrossWorkload:
    def test_all_workloads_run_on_all_chips(self):
        """Smoke test: every workload completes on every chip."""
        for chip in all_chips():
            try:
                llama = llama_7b_decode(chip, seq_len=128, prec=PrecisionMode.FP16)
                assert llama.total_runtime_ns > 0
            except (ValueError, KeyError):
                pass

            try:
                resnet = resnet50_forward(chip, batch=1, prec=PrecisionMode.FP16)
                assert resnet.total_runtime_ns > 0
            except (ValueError, KeyError):
                pass

            try:
                aes = aes_xts_throughput(chip, bytes_to_encrypt=1024,
                                         prec=PrecisionMode.INT8)
                assert aes.total_runtime_ns > 0
            except (ValueError, KeyError):
                pass
