// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: pairwise / tree summation reductions.
//
// Three flavours of summation are exposed for benchmarking:
//
//   * naive_sum      - simple left-to-right accumulator. Worst case
//                      relative error grows as O(n * eps).
//   * pairwise_sum   - recursive halving; relative error is O(log n * eps).
//   * kahan_sum      - compensated summation, relative error is O(eps)
//                      regardless of n. Provided for high-accuracy modes.
//
// All three operate on a contiguous buffer and are templated on the
// floating point type. For benchmarks we expose both the recursive and
// the iterative formulations of pairwise summation; they produce the same
// answer up to floating point rounding (which is tested explicitly).

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

namespace zhilicon::kernels::reduce {

// Plain left-to-right sum. Used as the baseline for accuracy comparisons.
template <typename T>
T naive_sum(const T* data, std::size_t length) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "naive_sum requires a floating point T");
  T acc = T{0};
  for (std::size_t i = 0; i < length; ++i) {
    acc += data[i];
  }
  return acc;
}

// Compensated (Kahan) summation. Highest accuracy of the three variants.
template <typename T>
T kahan_sum(const T* data, std::size_t length) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "kahan_sum requires a floating point T");
  T sum = T{0};
  T c = T{0};
  for (std::size_t i = 0; i < length; ++i) {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

namespace detail {

// Recursive pairwise summation with a base case to bound recursion depth
// at log2(length). The base case threshold is chosen so the inner loop
// fits comfortably in the L1 cache of any modern CPU; tests verify that
// the threshold does not change the result modulo rounding.
template <typename T>
T pairwise_sum_impl(const T* data, std::size_t length) noexcept {
  constexpr std::size_t base_case = 16;
  if (length <= base_case) {
    T acc = T{0};
    for (std::size_t i = 0; i < length; ++i) {
      acc += data[i];
    }
    return acc;
  }
  std::size_t half = length / 2;
  T left = pairwise_sum_impl<T>(data, half);
  T right = pairwise_sum_impl<T>(data + half, length - half);
  return left + right;
}

}  // namespace detail

// Pairwise (binary tree) summation. Provides numerical stability without
// the full overhead of Kahan summation. This is what numpy / Eigen use as
// the default for floating point reductions.
template <typename T>
T pairwise_sum(const T* data, std::size_t length) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "pairwise_sum requires a floating point T");
  if (length == 0) {
    return T{0};
  }
  return detail::pairwise_sum_impl<T>(data, length);
}

// Iterative variant of pairwise summation, useful when recursion is
// undesirable (e.g. in environments with constrained stacks). Produces
// the same value as pairwise_sum up to floating point rounding.
template <typename T>
T pairwise_sum_iterative(const T* data, std::size_t length) {
  static_assert(std::is_floating_point<T>::value,
                "pairwise_sum_iterative requires a floating point T");
  if (length == 0) {
    return T{0};
  }
  std::vector<T> level(data, data + length);
  while (level.size() > 1) {
    std::size_t pairs = level.size() / 2;
    for (std::size_t i = 0; i < pairs; ++i) {
      level[i] = level[2 * i] + level[2 * i + 1];
    }
    if (level.size() % 2 == 1) {
      level[pairs] = level[level.size() - 1];
      level.resize(pairs + 1);
    } else {
      level.resize(pairs);
    }
  }
  return level[0];
}

// Mean of a contiguous buffer using pairwise summation under the hood.
template <typename T>
T pairwise_mean(const T* data, std::size_t length) noexcept {
  if (length == 0) {
    return T{0};
  }
  return pairwise_sum<T>(data, length) / static_cast<T>(length);
}

}  // namespace zhilicon::kernels::reduce
