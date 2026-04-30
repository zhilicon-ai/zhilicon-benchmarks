// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../quant/quantize.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace zhilicon::kernels::quant;

ZH_TEST(symmetric_params_zero_signal) {
  std::vector<float> data = {0.0f, 0.0f, 0.0f};
  auto p = compute_symmetric_params(data.data(), data.size());
  ZH_CHECK_EQ(p.zero_point, 0);
  ZH_CHECK_NEAR(p.scale, 1.0f, 0.0f);
}

ZH_TEST(symmetric_params_basic) {
  std::vector<float> data = {-1.0f, 0.5f, 0.75f, 0.0f, 0.0f};
  auto p = compute_symmetric_params(data.data(), data.size());
  ZH_CHECK_EQ(p.zero_point, 0);
  // max abs = 1.0 -> scale = 1.0 / 127.0
  ZH_CHECK_NEAR(p.scale, 1.0f / 127.0f, 1e-6);
}

ZH_TEST(symmetric_quantize_round_trip) {
  std::vector<float> data = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  auto p = compute_symmetric_params(data.data(), data.size());
  std::vector<std::int8_t> q(data.size());
  quantize_symmetric(data.data(), data.size(), p, q.data());
  std::vector<float> dq(data.size());
  dequantize(q.data(), q.size(), p, dq.data());
  for (std::size_t i = 0; i < data.size(); ++i) {
    ZH_CHECK_NEAR(dq[i], data[i], 0.02f);
  }
}

ZH_TEST(symmetric_quantize_clip_at_extremes) {
  std::vector<float> data = {-2.0f, 2.0f};
  QuantParams p;
  p.scale = 1.0f / 127.0f;
  p.zero_point = 0;
  std::vector<std::int8_t> q(2);
  quantize_symmetric(data.data(), 2, p, q.data());
  ZH_CHECK_EQ(static_cast<int>(q[0]), -128);
  ZH_CHECK_EQ(static_cast<int>(q[1]), 127);
}

ZH_TEST(asymmetric_params_zero_signal) {
  std::vector<float> data = {0.0f, 0.0f};
  auto p = compute_asymmetric_params(data.data(), data.size());
  ZH_CHECK_EQ(p.zero_point, 0);
  ZH_CHECK_NEAR(p.scale, 1.0f, 0.0f);
}

ZH_TEST(asymmetric_params_positive_only) {
  // Values 0 to 10. Zero point must include 0 in the mapped range.
  std::vector<float> data = {0.0f, 5.0f, 10.0f};
  auto p = compute_asymmetric_params(data.data(), data.size());
  // range = 10 -> scale = 10/255
  ZH_CHECK_NEAR(p.scale, 10.0f / 255.0f, 1e-6);
}

ZH_TEST(asymmetric_quantize_round_trip) {
  std::vector<float> data;
  std::mt19937 rng(31);
  std::uniform_real_distribution<float> dist(-3.0f, 5.0f);
  for (std::size_t i = 0; i < 64; ++i) data.push_back(dist(rng));
  auto p = compute_asymmetric_params(data.data(), data.size());
  std::vector<std::int8_t> q(data.size());
  quantize_asymmetric(data.data(), data.size(), p, q.data());
  std::vector<float> dq(data.size());
  dequantize(q.data(), q.size(), p, dq.data());
  // The reconstruction error should be at most one scale step.
  for (std::size_t i = 0; i < data.size(); ++i) {
    ZH_CHECK_NEAR(dq[i], data[i], p.scale);
  }
}

ZH_TEST(asymmetric_quantize_endpoints_map_correctly) {
  std::vector<float> data = {-3.0f, 5.0f};
  auto p = compute_asymmetric_params(data.data(), data.size());
  std::vector<std::int8_t> q(2);
  quantize_asymmetric(data.data(), 2, p, q.data());
  // Min should map near -128, max near 127.
  ZH_CHECK_EQ(static_cast<int>(q[0]), -128);
  ZH_CHECK_EQ(static_cast<int>(q[1]), 127);
}

ZH_TEST(asymmetric_dequantize_zero_returns_neg_zp_scale) {
  // Dequantizing the integer q == 0 returns -zp * scale.
  QuantParams p;
  p.scale = 0.5f;
  p.zero_point = -10;
  std::int8_t q = 0;
  float dq = 0.0f;
  dequantize(&q, 1, p, &dq);
  ZH_CHECK_NEAR(dq, 5.0f, 1e-6);
}

ZH_TEST(symmetric_zero_input_gives_zero) {
  QuantParams p;
  p.scale = 0.1f;
  p.zero_point = 0;
  float zero = 0.0f;
  std::int8_t q = 0;
  quantize_symmetric(&zero, 1, p, &q);
  ZH_CHECK_EQ(static_cast<int>(q), 0);
}

ZH_TEST(asymmetric_constant_signal) {
  std::vector<float> data = {2.0f, 2.0f, 2.0f};
  auto p = compute_asymmetric_params(data.data(), data.size());
  // lo = 0 (due to forced 0-inclusion), hi = 2
  ZH_CHECK_NEAR(p.scale, 2.0f / 255.0f, 1e-6);
}

ZH_TEST(symmetric_round_to_even_half_case) {
  // Verify round-half-to-even: 0.5 -> 0 (even), 1.5 -> 2 (even).
  // We construct a scale where the input is exactly on a half boundary.
  QuantParams p;
  p.scale = 1.0f;
  p.zero_point = 0;
  float vals[] = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f};
  std::int8_t q[5];
  quantize_symmetric(vals, 5, p, q);
  ZH_CHECK_EQ(static_cast<int>(q[0]), 0);
  ZH_CHECK_EQ(static_cast<int>(q[1]), 2);
  ZH_CHECK_EQ(static_cast<int>(q[2]), 2);
  ZH_CHECK_EQ(static_cast<int>(q[3]), 0);
  ZH_CHECK_EQ(static_cast<int>(q[4]), -2);
}

ZH_TEST(asymmetric_negative_only_clamps_lo) {
  // All values negative; lo should be clamped to 0 to keep zero in range.
  std::vector<float> data = {-10.0f, -5.0f, -1.0f};
  auto p = compute_asymmetric_params(data.data(), data.size());
  // After clamp, lo = -10, hi = 0, range = 10.
  ZH_CHECK_NEAR(p.scale, 10.0f / 255.0f, 1e-6);
}

ZH_TEST(asymmetric_empty_input) {
  auto p = compute_asymmetric_params(nullptr, 0);
  ZH_CHECK_EQ(p.zero_point, 0);
  ZH_CHECK_NEAR(p.scale, 1.0f, 0.0f);
}

ZH_TEST_MAIN("quant/quantize")
