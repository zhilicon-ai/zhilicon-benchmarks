"""Tests for the CLI -- exercise every subcommand and check exit codes + stdout."""

from __future__ import annotations

import io
import json
import sys
import pytest

from perf_models.cli import main


# Helper to capture stdout
class _CaptureStdout:
    def __enter__(self):
        self._old = sys.stdout
        self._buf = io.StringIO()
        sys.stdout = self._buf
        return self

    def __exit__(self, *args):
        sys.stdout = self._old

    @property
    def value(self) -> str:
        return self._buf.getvalue()


class TestSummary:
    def test_summary_one_chip(self):
        with _CaptureStdout() as cap:
            rc = main(["summary", "--chip", "Sentinel-1"])
        assert rc == 0
        assert "Sentinel-1" in cap.value
        assert "TOPS/W" in cap.value

    def test_summary_all(self):
        with _CaptureStdout() as cap:
            rc = main(["summary", "--chip", "all"])
        assert rc == 0
        for name in ["Sentinel-1", "Horizon-1", "Discovery-1", "Nexus-1", "Prometheus"]:
            assert name in cap.value

    def test_summary_csv(self):
        with _CaptureStdout() as cap:
            rc = main(["summary", "--chip", "Sentinel-1,Discovery-1"])
        assert rc == 0
        assert "Sentinel-1" in cap.value
        assert "Discovery-1" in cap.value


class TestKernel:
    def test_matmul_default(self):
        with _CaptureStdout() as cap:
            rc = main(["kernel", "matmul", "--chip", "Discovery-1"])
        assert rc == 0
        assert "matmul" in cap.value

    def test_matmul_custom_shape(self):
        with _CaptureStdout() as cap:
            rc = main(["kernel", "matmul", "--chip", "Sentinel-1",
                       "--m", "256", "--n", "256", "--k", "256"])
        assert rc == 0
        assert "matmul_256x256x256" in cap.value

    def test_attention(self):
        with _CaptureStdout() as cap:
            rc = main(["kernel", "attention", "--chip", "Prometheus",
                       "--b", "1", "--h-heads", "32", "--s-seq", "512", "--d", "128"])
        assert rc == 0
        assert "attention" in cap.value

    def test_conv2d(self):
        with _CaptureStdout() as cap:
            rc = main(["kernel", "conv2d", "--chip", "Discovery-1",
                       "--n-batch", "1", "--c", "3", "--h", "224", "--w", "224",
                       "--k", "64", "--r", "7", "--s", "7"])
        assert rc == 0
        assert "conv2d" in cap.value

    def test_fft(self):
        with _CaptureStdout() as cap:
            rc = main(["kernel", "fft", "--chip", "Nexus-1", "--n-fft", "256"])
        assert rc == 0
        assert "fft_256" in cap.value

    def test_gemv(self):
        with _CaptureStdout() as cap:
            rc = main(["kernel", "gemv", "--chip", "Sentinel-1",
                       "--m", "1024", "--n", "1024"])
        assert rc == 0
        assert "gemv_1024x1024" in cap.value

    def test_invalid_precision(self):
        with pytest.raises(SystemExit):
            main(["kernel", "matmul", "--chip", "Sentinel-1", "--prec", "bogus"])


class TestWorkload:
    def test_llama(self):
        with _CaptureStdout() as cap:
            rc = main(["workload", "llama", "--chip", "Prometheus",
                       "--seq-len", "128"])
        assert rc == 0
        assert "LLaMA" in cap.value

    def test_resnet50(self):
        with _CaptureStdout() as cap:
            rc = main(["workload", "resnet50", "--chip", "Discovery-1",
                       "--batch", "1"])
        assert rc == 0
        assert "ResNet" in cap.value

    def test_aes(self):
        with _CaptureStdout() as cap:
            rc = main(["workload", "aes", "--chip", "Sentinel-1",
                       "--bytes", "1048576", "--prec", "int8"])
        assert rc == 0
        assert "AES" in cap.value

    def test_ofdm(self):
        with _CaptureStdout() as cap:
            rc = main(["workload", "ofdm256", "--chip", "Nexus-1"])
        assert rc == 0
        assert "OFDM" in cap.value

    def test_kyber(self):
        with _CaptureStdout() as cap:
            rc = main(["workload", "kyber768", "--chip", "Sentinel-1",
                       "--prec", "int16"])
        assert rc == 0
        assert "Kyber" in cap.value

    def test_dicom(self):
        with _CaptureStdout() as cap:
            rc = main(["workload", "dicom", "--chip", "Discovery-1",
                       "--volume", "64"])
        assert rc == 0
        assert "DICOM" in cap.value


class TestPower:
    def test_power_specific_op(self):
        with _CaptureStdout() as cap:
            rc = main(["power", "--chip", "Sentinel-1", "--op", "nominal"])
        assert rc == 0
        assert "TOTAL" in cap.value

    def test_power_sweep(self):
        with _CaptureStdout() as cap:
            rc = main(["power", "--chip", "Sentinel-1"])
        assert rc == 0
        # Sweep should produce one breakdown per op (3 for Sentinel)
        assert cap.value.count("=== Power breakdown") == 3


class TestThermal:
    def test_thermal_low_power(self):
        with _CaptureStdout() as cap:
            rc = main(["thermal", "--chip", "Sentinel-1", "--power", "10"])
        assert rc == 0
        assert "Steady-state Tj" in cap.value
        assert "no" in cap.value or "never" in cap.value  # Should not throttle

    def test_thermal_high_power_throttles(self):
        with _CaptureStdout() as cap:
            rc = main(["thermal", "--chip", "Sentinel-1", "--power", "300"])
        assert rc == 0
        assert "YES" in cap.value
        assert "Time to throttle" in cap.value


class TestCompare:
    def test_compare_llama(self):
        with _CaptureStdout() as cap:
            rc = main(["compare", "--chips", "Sentinel-1,Prometheus",
                       "--workload", "llama", "--seq-len", "128"])
        assert rc == 0
        assert "Sentinel-1" in cap.value
        assert "Prometheus" in cap.value

    def test_compare_matmul(self):
        with _CaptureStdout() as cap:
            rc = main(["compare", "--chips", "Sentinel-1,Discovery-1",
                       "--workload", "matmul", "--m", "256", "--n", "256", "--k", "256"])
        assert rc == 0
        assert "matmul" in cap.value


class TestOptimalOp:
    def test_for_one_chip(self):
        with _CaptureStdout() as cap:
            rc = main(["optimal-op", "--chip", "Discovery-1"])
        assert rc == 0
        assert "optimal op" in cap.value

    def test_for_all_chips(self):
        with _CaptureStdout() as cap:
            rc = main(["optimal-op", "--chip", "all"])
        assert rc == 0
        # 5 chips, but skip the ones missing FP16 in the precision list
        assert cap.value.count("optimal op") >= 4


class TestExportJson:
    def test_export_one(self):
        with _CaptureStdout() as cap:
            rc = main(["export-json", "--chip", "Sentinel-1"])
        assert rc == 0
        # Must be valid JSON
        d = json.loads(cap.value)
        assert "Sentinel-1" in d
        assert d["Sentinel-1"]["die_area_mm2"] == 112.0

    def test_export_all(self):
        with _CaptureStdout() as cap:
            rc = main(["export-json", "--chip", "all"])
        assert rc == 0
        d = json.loads(cap.value)
        for name in ["Sentinel-1", "Horizon-1", "Discovery-1", "Nexus-1", "Prometheus"]:
            assert name in d


class TestErrorHandling:
    def test_unknown_chip(self):
        with pytest.raises((SystemExit, ValueError)):
            main(["summary", "--chip", "MagicChip"])

    def test_unknown_op(self):
        with pytest.raises(SystemExit):
            main(["kernel", "matmul", "--chip", "Sentinel-1", "--op", "nonexistent"])

    def test_no_subcommand(self):
        with pytest.raises(SystemExit):
            main([])
