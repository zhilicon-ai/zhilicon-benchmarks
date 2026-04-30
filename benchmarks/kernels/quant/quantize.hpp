// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: float -> int8 quantization (symmetric & asymmetric).
//
// Two flavours are supported:
//
//   * Symmetric quantization: q = clip(round(x / scale), -127, 127). The
//     zero point is fixed at 0. This matches the convention used by the
//     PyTorch quantization tooling for weights.
//
//   * Asymmetric quantization: q = clip(round(x / scale + zp), -128, 127).
//     The zero point is chosen so that the input range is mapped onto the
//     full int8 range. This matches the convention used for activations.
//
// Both routines compute the parameters from a per-tensor min/max scan and
// the corresponding dequantize routines invert the quantization. Round-to-
// nearest, ties-to-even is used to match TensorFlow / ONNXRuntime.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace zhilicon::kernels::quant {

struct QuantParams {
  float scale{1.0f};
  std::int32_t zero_point{0};
};

namespace detail {

// Round half to even. std::nearbyint with round-to-even is platform-dependent
// without an explicit fesetround; we provide a portable software version so
// behaviour is bit-stable across CI runners.
inline float round_half_to_even(float x) noexcept {
  float r = std::round(x);
  float diff = x - std::floor(x);
  if (diff == 0.5f || diff == -0.5f) {
    // Halfway case: round to even.
    float f = std::floor(x);
    if (std::fmod(f, 2.0f) == 0.0f) {
      r = f;
    } else {
      r = f + 1.0f;
    }
  }
  return r;
}

inline std::int8_t clip_to_int8(float v) noexcept {
  if (v >= 127.0f) {
    return std::numeric_limits<std::int8_t>::max();
  }
  if (v <= -128.0f) {
    return std::numeric_limits<std::int8_t>::min();
  }
  return static_cast<std::int8_t>(v);
}

}  // namespace detail

// Compute symmetric quantization parameters. zero_point is forced to 0 and
// scale is chosen so that |x|_max maps to 127.
inline QuantParams compute_symmetric_params(const float* data,
                                            std::size_t length) noexcept {
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < length; ++i) {
    float a = std::fabs(data[i]);
    if (a > max_abs) {
      max_abs = a;
    }
  }
  QuantParams q;
  q.zero_point = 0;
  if (max_abs == 0.0f) {
    q.scale = 1.0f;
  } else {
    q.scale = max_abs / 127.0f;
  }
  return q;
}

// Compute asymmetric quantization parameters that map [min, max] -> [-128, 127].
inline QuantParams compute_asymmetric_params(const float* data,
                                             std::size_t length) noexcept {
  if (length == 0) {
    return QuantParams{};
  }
  float lo = data[0];
  float hi = data[0];
  for (std::size_t i = 1; i < length; ++i) {
    if (data[i] < lo) {
      lo = data[i];
    }
    if (data[i] > hi) {
      hi = data[i];
    }
  }
  // Always include 0 in the range, otherwise the zero-point is undefined.
  if (lo > 0.0f) {
    lo = 0.0f;
  }
  if (hi < 0.0f) {
    hi = 0.0f;
  }
  float range = hi - lo;
  QuantParams q;
  if (range == 0.0f) {
    q.scale = 1.0f;
    q.zero_point = 0;
    return q;
  }
  q.scale = range / 255.0f;
  // zero_point in int8 range. -128 corresponds to lo, 127 to hi.
  float zp_float = -128.0f - lo / q.scale;
  zp_float = std::round(zp_float);
  if (zp_float < -128.0f) {
    zp_float = -128.0f;
  } else if (zp_float > 127.0f) {
    zp_float = 127.0f;
  }
  q.zero_point = static_cast<std::int32_t>(zp_float);
  return q;
}

// Apply symmetric quantization. zero_point is assumed to be 0.
inline void quantize_symmetric(const float* in, std::size_t length,
                               const QuantParams& params,
                               std::int8_t* out) noexcept {
  float inv_scale = 1.0f / params.scale;
  for (std::size_t i = 0; i < length; ++i) {
    float v = detail::round_half_to_even(in[i] * inv_scale);
    out[i] = detail::clip_to_int8(v);
  }
}

// Apply asymmetric quantization with a possibly non-zero zero_point.
inline void quantize_asymmetric(const float* in, std::size_t length,
                                const QuantParams& params,
                                std::int8_t* out) noexcept {
  float inv_scale = 1.0f / params.scale;
  for (std::size_t i = 0; i < length; ++i) {
    float v = detail::round_half_to_even(in[i] * inv_scale)
              + static_cast<float>(params.zero_point);
    out[i] = detail::clip_to_int8(v);
  }
}

// Dequantize int8 -> float, common implementation for both symmetric and
// asymmetric (the params struct carries the correct zero_point).
inline void dequantize(const std::int8_t* in, std::size_t length,
                       const QuantParams& params, float* out) noexcept {
  for (std::size_t i = 0; i < length; ++i) {
    out[i] = (static_cast<float>(in[i]) - static_cast<float>(params.zero_point))
             * params.scale;
  }
}

}  // namespace zhilicon::kernels::quant
