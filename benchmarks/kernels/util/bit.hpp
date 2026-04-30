// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: portable bit manipulation primitives.
//
// Many kernels in the suite (FFT bit-reversal, popcount-style reductions,
// alignment checks) need fast bit primitives. We expose constexpr-friendly
// fallbacks plus thin wrappers around compiler intrinsics where available.
// The fallbacks are exercised by the unit tests so coverage is portable.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace zhilicon::kernels::util {

// Population count (number of set bits). Constexpr fallback that works for
// any unsigned integer type up to 64 bits.
template <typename T>
constexpr int popcount(T x) noexcept {
  static_assert(std::is_unsigned<T>::value, "popcount requires unsigned type");
  int count = 0;
  while (x != 0) {
    count += static_cast<int>(x & T{1});
    x >>= 1;
  }
  return count;
}

// Count trailing zeros. Returns the bit-width of T when x == 0, matching
// the convention used by std::countr_zero in C++20.
template <typename T>
constexpr int ctz(T x) noexcept {
  static_assert(std::is_unsigned<T>::value, "ctz requires unsigned type");
  if (x == 0) {
    return std::numeric_limits<T>::digits;
  }
  int count = 0;
  while ((x & T{1}) == 0) {
    ++count;
    x >>= 1;
  }
  return count;
}

// Count leading zeros. Returns the bit-width of T when x == 0.
template <typename T>
constexpr int clz(T x) noexcept {
  static_assert(std::is_unsigned<T>::value, "clz requires unsigned type");
  constexpr int bits = std::numeric_limits<T>::digits;
  if (x == 0) {
    return bits;
  }
  int count = 0;
  T mask = T{1} << (bits - 1);
  while ((x & mask) == 0) {
    ++count;
    mask >>= 1;
  }
  return count;
}

// Bit-reverse n bits of x. Used by radix-2 FFT permutation.
constexpr std::uint32_t bit_reverse(std::uint32_t x, int bits) noexcept {
  std::uint32_t out = 0;
  for (int i = 0; i < bits; ++i) {
    out = (out << 1) | (x & 1u);
    x >>= 1;
  }
  return out;
}

// Returns true if x is a power of two and non-zero.
template <typename T>
constexpr bool is_pow2(T x) noexcept {
  static_assert(std::is_unsigned<T>::value, "is_pow2 requires unsigned type");
  return x != 0 && (x & (x - 1)) == 0;
}

// Compute the smallest power of two >= x. Returns 0 if the result would
// overflow T.
template <typename T>
constexpr T next_pow2(T x) noexcept {
  static_assert(std::is_unsigned<T>::value, "next_pow2 requires unsigned type");
  if (x <= 1) {
    return T{1};
  }
  // Decrement and find highest bit.
  T n = x - 1;
  for (int shift = 1; shift < std::numeric_limits<T>::digits; shift <<= 1) {
    n |= n >> shift;
  }
  // Overflow check: if all bits were already set, +1 wraps to zero.
  if (n == std::numeric_limits<T>::max()) {
    return 0;
  }
  return n + 1;
}

// Integer log2 of a power-of-two input. Returns -1 for non-powers of two
// and 0 for x == 1. Useful for FFT length computations where the input is
// validated to be a power of two upstream.
template <typename T>
constexpr int log2_pow2(T x) noexcept {
  static_assert(std::is_unsigned<T>::value, "log2_pow2 requires unsigned type");
  if (!is_pow2(x)) {
    return -1;
  }
  int n = 0;
  while ((x >> n) > 1) {
    ++n;
  }
  return n;
}

}  // namespace zhilicon::kernels::util
