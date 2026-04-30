// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: software fp16 (IEEE 754 binary16) <-> fp32 conversion.
//
// The benchmark suite intentionally does NOT depend on hardware fp16
// intrinsics (__fp16, _Float16, _cvtss_sh) so that simulator hosts and
// CI runners without ARMv8.2-FP or F16C extensions can still exercise
// quantization-related kernels. The encoding is stored in a uint16_t.
//
// The conversion is the canonical Mike Acton routine extended to handle
// subnormals and signed zero exactly. Bit-exact corner cases are tested
// in half_test.cpp.

#pragma once

#include <cstdint>
#include <cstring>

namespace zhilicon::kernels::util {

// 16-bit IEEE half-precision float, stored as raw bits.
struct half {
  std::uint16_t bits;

  constexpr half() noexcept : bits(0) {}
  constexpr explicit half(std::uint16_t b) noexcept : bits(b) {}
};

// Helper: bitcast float <-> uint32_t. C++20 has std::bit_cast; we stay on
// C++17 with memcpy which is the canonical UB-free spelling.
inline std::uint32_t float_to_bits(float f) noexcept {
  std::uint32_t out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline float bits_to_float(std::uint32_t u) noexcept {
  float out;
  std::memcpy(&out, &u, sizeof(out));
  return out;
}

// Convert fp32 -> fp16. Round-to-nearest-even, with infinities and NaNs
// preserved. Sub-fp16-normal inputs become fp16 subnormals or signed zero.
inline half float_to_half(float f) noexcept {
  std::uint32_t x = float_to_bits(f);
  std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  std::int32_t exp32 = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127;
  std::uint32_t mant32 = x & 0x007FFFFFu;

  // NaN or infinity in fp32.
  if (exp32 == 128) {
    if (mant32 == 0) {
      // Infinity -> fp16 infinity.
      return half(static_cast<std::uint16_t>(sign | 0x7C00u));
    }
    // NaN -> quiet fp16 NaN, preserve top bit of payload.
    std::uint16_t payload = static_cast<std::uint16_t>(mant32 >> 13);
    if (payload == 0) {
      payload = 1;  // ensure NaN, not infinity
    }
    return half(static_cast<std::uint16_t>(sign | 0x7C00u | payload));
  }

  // Bias to fp16. fp16 bias is 15.
  std::int32_t exp16 = exp32 + 15;

  if (exp16 >= 31) {
    // Overflow -> infinity.
    return half(static_cast<std::uint16_t>(sign | 0x7C00u));
  }

  if (exp16 <= 0) {
    // Subnormal or underflow to zero. We need to shift the mantissa
    // (with the implicit leading 1) right by (1 - exp16) bits.
    if (exp16 < -10) {
      // Even with subnormal representation the value rounds to zero.
      return half(sign);
    }
    std::uint32_t mant = mant32 | 0x00800000u;
    int shift = 14 - exp16;  // 14 = 23 - 10 + 1
    std::uint32_t round_bit = (mant >> (shift - 1)) & 1u;
    std::uint32_t sticky_mask = (shift >= 2) ? ((1u << (shift - 1)) - 1u) : 0u;
    std::uint32_t sticky = (mant & sticky_mask) ? 1u : 0u;
    std::uint16_t mant16 = static_cast<std::uint16_t>(mant >> shift);
    if (round_bit && (sticky || (mant16 & 1u))) {
      ++mant16;
    }
    return half(static_cast<std::uint16_t>(sign | mant16));
  }

  // Normal number. Round mantissa from 23 bits to 10 bits.
  std::uint32_t round_bit = (mant32 >> 12) & 1u;
  std::uint32_t sticky = (mant32 & 0x00000FFFu) ? 1u : 0u;
  std::uint16_t mant16 = static_cast<std::uint16_t>(mant32 >> 13);
  std::uint16_t exp_field = static_cast<std::uint16_t>(exp16);
  std::uint16_t out = static_cast<std::uint16_t>(sign | (exp_field << 10) | mant16);
  if (round_bit && (sticky || (out & 1u))) {
    ++out;
    // If rounding overflows the mantissa into the exponent and produces
    // infinity, that is the correct IEEE behaviour.
  }
  return half(out);
}

// Convert fp16 -> fp32. Lossless on fp16 inputs other than NaN payload bits.
inline float half_to_float(half h) noexcept {
  std::uint32_t x = h.bits;
  std::uint32_t sign = (x & 0x8000u) << 16;
  std::uint32_t exp = (x >> 10) & 0x1Fu;
  std::uint32_t mant = x & 0x03FFu;

  std::uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      // Signed zero.
      out = sign;
    } else {
      // Subnormal. Normalise by shifting until the implicit leading 1 is
      // in bit 10 (the position of the hidden bit for fp16 normals); the
      // number of shifts (k) determines the binade. The fp16 subnormal
      // value M * 2^-24 maps to fp32 with mantissa-shifts = (1 + k) and
      // exponent (127 - 14 - k).
      std::uint32_t k = 0;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        ++k;
      }
      mant &= 0x03FFu;
      out = sign | ((127u - 14u - k) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    // Infinity or NaN.
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    // Normal.
    out = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  return bits_to_float(out);
}

// Returns true if h represents a NaN.
inline bool is_nan(half h) noexcept {
  return ((h.bits & 0x7C00u) == 0x7C00u) && ((h.bits & 0x03FFu) != 0);
}

// Returns true if h represents +Inf or -Inf.
inline bool is_inf(half h) noexcept {
  return ((h.bits & 0x7C00u) == 0x7C00u) && ((h.bits & 0x03FFu) == 0);
}

}  // namespace zhilicon::kernels::util
