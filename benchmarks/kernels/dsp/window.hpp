// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: window functions for FFT pre-processing.
//
// Provides the canonical windows used by the spectral analysis kernels
// in the benchmark suite: Hann, Hamming, Blackman, and rectangular. Each
// function fills a caller-provided buffer of length N with the window
// values, indexed 0 .. N-1. The conventions match scipy.signal.windows.

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace zhilicon::kernels::dsp {

namespace detail {
constexpr double kPiWindow = 3.14159265358979323846;
}

// Rectangular (boxcar) window. All samples set to 1.0.
template <typename T>
void rectangular_window(T* w, std::size_t n) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "rectangular_window requires a floating point T");
  for (std::size_t i = 0; i < n; ++i) {
    w[i] = T{1};
  }
}

// Hann (raised cosine) window: w[i] = 0.5 * (1 - cos(2 pi i / (N-1))).
// For N == 1 we emit a single 1.0 sample to mirror scipy's convention.
template <typename T>
void hann_window(T* w, std::size_t n) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "hann_window requires a floating point T");
  if (n == 0) {
    return;
  }
  if (n == 1) {
    w[0] = T{1};
    return;
  }
  const double scale = 2.0 * detail::kPiWindow /
                       static_cast<double>(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    w[i] = static_cast<T>(0.5 * (1.0 - std::cos(scale * static_cast<double>(i))));
  }
}

// Hamming window: w[i] = 0.54 - 0.46 cos(2 pi i / (N-1)).
template <typename T>
void hamming_window(T* w, std::size_t n) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "hamming_window requires a floating point T");
  if (n == 0) {
    return;
  }
  if (n == 1) {
    w[0] = T{1};
    return;
  }
  const double scale = 2.0 * detail::kPiWindow /
                       static_cast<double>(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    w[i] = static_cast<T>(0.54 - 0.46 *
                                  std::cos(scale * static_cast<double>(i)));
  }
}

// Blackman window: 0.42 - 0.5 cos(2 pi i / (N-1)) + 0.08 cos(4 pi i / (N-1)).
template <typename T>
void blackman_window(T* w, std::size_t n) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "blackman_window requires a floating point T");
  if (n == 0) {
    return;
  }
  if (n == 1) {
    w[0] = T{1};
    return;
  }
  const double scale = 2.0 * detail::kPiWindow /
                       static_cast<double>(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    double phi = scale * static_cast<double>(i);
    w[i] = static_cast<T>(0.42 - 0.5 * std::cos(phi) +
                          0.08 * std::cos(2.0 * phi));
  }
}

// Apply a window in-place: out[i] = data[i] * w[i]. Output may alias data.
template <typename T>
void apply_window(const T* data, const T* w, std::size_t n, T* out) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = data[i] * w[i];
  }
}

}  // namespace zhilicon::kernels::dsp
