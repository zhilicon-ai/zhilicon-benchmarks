// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: radix-2 Cooley-Tukey FFT.
//
// In-place decimation-in-time FFT for power-of-two lengths. The
// implementation is templated on the floating point type so the same
// kernel can be benchmarked at fp32 and fp64 precision. The reference
// for correctness tests is the naive O(N^2) DFT in this same header.
//
// Twiddle factors are computed on the fly to keep the header self-
// contained. For benchmarks the caller can pre-compute them via the
// FftPlan helper, which caches the bit-reversal permutation and the
// twiddle table.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "../util/bit.hpp"

namespace zhilicon::kernels::dsp {

// Naive O(N^2) DFT used as a reference and as a slow path for non-power-of-two
// inputs in the test suite.
template <typename Real>
void naive_dft(const std::complex<Real>* input, std::size_t n,
               std::complex<Real>* output) noexcept {
  static_assert(std::is_floating_point<Real>::value,
                "naive_dft requires a floating point real type");
  const Real two_pi = static_cast<Real>(2.0 * 3.14159265358979323846);
  for (std::size_t k = 0; k < n; ++k) {
    std::complex<Real> acc(0, 0);
    for (std::size_t j = 0; j < n; ++j) {
      Real theta = -two_pi * static_cast<Real>(k) * static_cast<Real>(j)
                   / static_cast<Real>(n);
      std::complex<Real> w(std::cos(theta), std::sin(theta));
      acc += input[j] * w;
    }
    output[k] = acc;
  }
}

// Bit-reverse permute the input buffer in place. n must be a power of two.
template <typename Real>
void bit_reverse_permute(std::complex<Real>* data, std::size_t n) noexcept {
  using zhilicon::kernels::util::log2_pow2;
  int bits = log2_pow2<std::size_t>(n);
  if (bits < 0) {
    return;
  }
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t j = zhilicon::kernels::util::bit_reverse(i, bits);
    if (j > i) {
      std::swap(data[i], data[j]);
    }
  }
}

// In-place radix-2 Cooley-Tukey FFT. Returns true on success, false if
// n is not a power of two. The forward transform uses the engineering
// sign convention W = exp(-j * 2 * pi / N).
template <typename Real>
bool fft_radix2_inplace(std::complex<Real>* data, std::size_t n) noexcept {
  static_assert(std::is_floating_point<Real>::value,
                "fft_radix2_inplace requires a floating point real type");
  if (n == 0 || !zhilicon::kernels::util::is_pow2<std::size_t>(n)) {
    return false;
  }
  if (n == 1) {
    return true;
  }
  bit_reverse_permute(data, n);
  const Real two_pi = static_cast<Real>(2.0 * 3.14159265358979323846);
  for (std::size_t size = 2; size <= n; size <<= 1) {
    const std::size_t half = size >> 1;
    Real theta = -two_pi / static_cast<Real>(size);
    std::complex<Real> w_step(std::cos(theta), std::sin(theta));
    for (std::size_t base = 0; base < n; base += size) {
      std::complex<Real> w(1, 0);
      for (std::size_t j = 0; j < half; ++j) {
        std::complex<Real> t = w * data[base + j + half];
        std::complex<Real> u = data[base + j];
        data[base + j] = u + t;
        data[base + j + half] = u - t;
        w *= w_step;
      }
    }
  }
  return true;
}

// Inverse radix-2 FFT. Output is scaled by 1/N to match the standard
// definition X^{-1}[n] = 1/N * sum X[k] exp(+j * 2 * pi * k * n / N).
template <typename Real>
bool ifft_radix2_inplace(std::complex<Real>* data, std::size_t n) noexcept {
  if (n == 0 || !zhilicon::kernels::util::is_pow2<std::size_t>(n)) {
    return false;
  }
  // Conjugate, forward FFT, conjugate, scale.
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = std::conj(data[i]);
  }
  if (!fft_radix2_inplace(data, n)) {
    return false;
  }
  Real scale = Real{1} / static_cast<Real>(n);
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = std::conj(data[i]) * scale;
  }
  return true;
}

// Pre-computed plan: caches the twiddle table for repeated invocations,
// matching the API style of FFTW / pocketfft. The table is
// half-length-per-stage flattened: stage s owns 2^(s-1) twiddles.
template <typename Real>
class FftPlan {
 public:
  FftPlan() = default;
  explicit FftPlan(std::size_t n) { resize(n); }

  bool resize(std::size_t n) {
    if (!zhilicon::kernels::util::is_pow2<std::size_t>(n)) {
      return false;
    }
    n_ = n;
    twiddles_.clear();
    if (n < 2) {
      return true;
    }
    const Real two_pi = static_cast<Real>(2.0 * 3.14159265358979323846);
    for (std::size_t size = 2; size <= n; size <<= 1) {
      const std::size_t half = size >> 1;
      Real theta = -two_pi / static_cast<Real>(size);
      std::complex<Real> w(1, 0);
      std::complex<Real> w_step(std::cos(theta), std::sin(theta));
      for (std::size_t j = 0; j < half; ++j) {
        twiddles_.push_back(w);
        w *= w_step;
      }
    }
    return true;
  }

  std::size_t size() const noexcept { return n_; }

  bool execute(std::complex<Real>* data) const noexcept {
    if (n_ == 0 || !zhilicon::kernels::util::is_pow2<std::size_t>(n_)) {
      return false;
    }
    if (n_ == 1) {
      return true;
    }
    bit_reverse_permute(data, n_);
    std::size_t twiddle_offset = 0;
    for (std::size_t size = 2; size <= n_; size <<= 1) {
      const std::size_t half = size >> 1;
      for (std::size_t base = 0; base < n_; base += size) {
        for (std::size_t j = 0; j < half; ++j) {
          const std::complex<Real>& w = twiddles_[twiddle_offset + j];
          std::complex<Real> t = w * data[base + j + half];
          std::complex<Real> u = data[base + j];
          data[base + j] = u + t;
          data[base + j + half] = u - t;
        }
      }
      twiddle_offset += half;
    }
    return true;
  }

 private:
  std::size_t n_{0};
  std::vector<std::complex<Real>> twiddles_;
};

}  // namespace zhilicon::kernels::dsp
