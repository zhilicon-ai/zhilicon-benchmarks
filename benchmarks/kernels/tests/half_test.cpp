// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../util/half.hpp"

#include <cmath>
#include <cstdint>

using namespace zhilicon::kernels::util;

namespace {

half make_half(std::uint16_t bits) { return half(bits); }

}  // namespace

ZH_TEST(half_zero_round_trip) {
  ZH_CHECK_EQ(float_to_half(0.0f).bits, std::uint16_t{0x0000});
  ZH_CHECK_EQ(float_to_half(-0.0f).bits, std::uint16_t{0x8000});
  ZH_CHECK_NEAR(half_to_float(make_half(0x0000)), 0.0f, 0.0f);
  // Negative zero: bit-exact via signbit.
  float negz = half_to_float(make_half(0x8000));
  ZH_CHECK(negz == 0.0f && std::signbit(negz));
}

ZH_TEST(half_one_round_trip) {
  half h = float_to_half(1.0f);
  ZH_CHECK_EQ(h.bits, std::uint16_t{0x3C00});
  ZH_CHECK_NEAR(half_to_float(h), 1.0f, 0.0f);
  half hn = float_to_half(-1.0f);
  ZH_CHECK_EQ(hn.bits, std::uint16_t{0xBC00});
}

ZH_TEST(half_powers_of_two) {
  // 2.0 -> 0x4000, 4.0 -> 0x4400, 0.5 -> 0x3800, 0.25 -> 0x3400
  ZH_CHECK_EQ(float_to_half(2.0f).bits, std::uint16_t{0x4000});
  ZH_CHECK_EQ(float_to_half(4.0f).bits, std::uint16_t{0x4400});
  ZH_CHECK_EQ(float_to_half(0.5f).bits, std::uint16_t{0x3800});
  ZH_CHECK_EQ(float_to_half(0.25f).bits, std::uint16_t{0x3400});
  ZH_CHECK_NEAR(half_to_float(make_half(0x4000)), 2.0f, 0.0f);
  ZH_CHECK_NEAR(half_to_float(make_half(0x3800)), 0.5f, 0.0f);
}

ZH_TEST(half_max_normal) {
  // 65504.0 (max normal fp16) -> 0x7BFF.
  ZH_CHECK_EQ(float_to_half(65504.0f).bits, std::uint16_t{0x7BFF});
  ZH_CHECK_NEAR(half_to_float(make_half(0x7BFF)), 65504.0f, 0.0f);
}

ZH_TEST(half_overflow_to_inf) {
  // Anything > 65504 + half_ulp rounds to +Inf in fp16.
  ZH_CHECK_EQ(float_to_half(70000.0f).bits, std::uint16_t{0x7C00});
  ZH_CHECK_EQ(float_to_half(-70000.0f).bits, std::uint16_t{0xFC00});
  ZH_CHECK_EQ(float_to_half(1e30f).bits, std::uint16_t{0x7C00});
}

ZH_TEST(half_inf_round_trip) {
  ZH_CHECK_EQ(float_to_half(std::numeric_limits<float>::infinity()).bits,
              std::uint16_t{0x7C00});
  ZH_CHECK_EQ(float_to_half(-std::numeric_limits<float>::infinity()).bits,
              std::uint16_t{0xFC00});
  ZH_CHECK(std::isinf(half_to_float(make_half(0x7C00))));
  ZH_CHECK(std::isinf(half_to_float(make_half(0xFC00))));
  ZH_CHECK(is_inf(make_half(0x7C00)));
  ZH_CHECK(is_inf(make_half(0xFC00)));
  ZH_CHECK(!is_nan(make_half(0x7C00)));
}

ZH_TEST(half_nan_round_trip) {
  half qnan = float_to_half(std::numeric_limits<float>::quiet_NaN());
  ZH_CHECK(is_nan(qnan));
  ZH_CHECK(std::isnan(half_to_float(qnan)));
  ZH_CHECK(is_nan(make_half(0x7E00)));
  ZH_CHECK(!is_inf(make_half(0x7E00)));
}

ZH_TEST(half_smallest_normal) {
  // 2^-14 = 6.10352e-5 -> smallest fp16 normal, bits 0x0400.
  half h = float_to_half(static_cast<float>(std::ldexp(1.0, -14)));
  ZH_CHECK_EQ(h.bits, std::uint16_t{0x0400});
  float f = half_to_float(make_half(0x0400));
  ZH_CHECK_NEAR(f, static_cast<float>(std::ldexp(1.0, -14)), 0.0f);
}

ZH_TEST(half_subnormal_smallest) {
  // 2^-24 -> smallest fp16 subnormal, bits 0x0001.
  half h = float_to_half(static_cast<float>(std::ldexp(1.0, -24)));
  ZH_CHECK_EQ(h.bits, std::uint16_t{0x0001});
  ZH_CHECK_NEAR(half_to_float(make_half(0x0001)),
                static_cast<float>(std::ldexp(1.0, -24)), 0.0f);
}

ZH_TEST(half_subnormal_arbitrary) {
  // 3 * 2^-24 -> subnormal with bits 0x0003.
  float v = 3.0f * static_cast<float>(std::ldexp(1.0, -24));
  half h = float_to_half(v);
  ZH_CHECK_EQ(h.bits, std::uint16_t{0x0003});
  ZH_CHECK_NEAR(half_to_float(make_half(0x0003)), v, 0.0f);
}

ZH_TEST(half_round_trip_exact_values) {
  // Values exactly representable in fp16 should survive a round trip.
  const float exact_values[] = {0.0f,    1.0f,     -1.0f,   2.0f,
                                0.5f,    0.25f,    1.5f,    100.0f,
                                -200.0f, 1024.0f,  -512.0f};
  for (float v : exact_values) {
    half h = float_to_half(v);
    float f = half_to_float(h);
    ZH_CHECK_NEAR(f, v, 0.0f);
  }
}

ZH_TEST(half_round_trip_inexact_within_tol) {
  // Values not exactly representable should round trip within fp16 ULP.
  const float values[] = {0.1f, 0.3f, 1.0f / 3.0f, 3.14159265f, -2.71828f};
  for (float v : values) {
    float f = half_to_float(float_to_half(v));
    // fp16 has ~3 decimal digits of precision; allow 1e-3 relative.
    float tol = std::fabs(v) * 1e-3f + 1e-7f;
    ZH_CHECK_NEAR(f, v, tol);
  }
}

ZH_TEST(half_underflow_to_zero) {
  // 2^-30 is well below the smallest fp16 subnormal so flushes to zero.
  half h = float_to_half(static_cast<float>(std::ldexp(1.0, -30)));
  ZH_CHECK_EQ(h.bits, std::uint16_t{0x0000});
}

ZH_TEST_MAIN("util/half")
