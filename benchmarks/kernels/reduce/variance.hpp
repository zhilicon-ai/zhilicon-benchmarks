// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: numerically stable variance / standard deviation.
//
// Welford's online algorithm avoids the catastrophic cancellation that
// affects the textbook (E[X^2] - E[X]^2) formula when the mean is large
// compared to the standard deviation. We expose:
//
//   * online_mean_var      - single-pass update returning mean and var
//   * sample_variance      - convenience wrapper using N-1 denominator
//   * population_variance  - convenience wrapper using N denominator

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace zhilicon::kernels::reduce {

template <typename T>
struct MeanVar {
  T mean{};
  T m2{};        // sum of squared deviations from the mean
  std::size_t count{0};
};

// Update an online accumulator with a new sample.
template <typename T>
void mean_var_update(MeanVar<T>& acc, T x) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "mean_var_update requires a floating point T");
  ++acc.count;
  T delta = x - acc.mean;
  acc.mean += delta / static_cast<T>(acc.count);
  T delta2 = x - acc.mean;
  acc.m2 += delta * delta2;
}

// Single-pass computation of mean and m2 across a buffer.
template <typename T>
MeanVar<T> online_mean_var(const T* data, std::size_t n) noexcept {
  MeanVar<T> acc;
  for (std::size_t i = 0; i < n; ++i) {
    mean_var_update(acc, data[i]);
  }
  return acc;
}

// Sample variance with N-1 denominator. Returns 0 for n < 2.
template <typename T>
T sample_variance(const T* data, std::size_t n) noexcept {
  if (n < 2) {
    return T{0};
  }
  auto acc = online_mean_var<T>(data, n);
  return acc.m2 / static_cast<T>(n - 1);
}

// Population variance with N denominator. Returns 0 for n == 0.
template <typename T>
T population_variance(const T* data, std::size_t n) noexcept {
  if (n == 0) {
    return T{0};
  }
  auto acc = online_mean_var<T>(data, n);
  return acc.m2 / static_cast<T>(n);
}

template <typename T>
T sample_stddev(const T* data, std::size_t n) noexcept {
  return std::sqrt(sample_variance<T>(data, n));
}

template <typename T>
T population_stddev(const T* data, std::size_t n) noexcept {
  return std::sqrt(population_variance<T>(data, n));
}

// Combine two MeanVar accumulators; useful for parallel reductions.
template <typename T>
MeanVar<T> merge(const MeanVar<T>& a, const MeanVar<T>& b) noexcept {
  if (a.count == 0) {
    return b;
  }
  if (b.count == 0) {
    return a;
  }
  MeanVar<T> out;
  out.count = a.count + b.count;
  T delta = b.mean - a.mean;
  out.mean = a.mean + delta * static_cast<T>(b.count) /
                          static_cast<T>(out.count);
  out.m2 = a.m2 + b.m2 + delta * delta * static_cast<T>(a.count) *
                            static_cast<T>(b.count) /
                            static_cast<T>(out.count);
  return out;
}

}  // namespace zhilicon::kernels::reduce
