// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../dsp/fir.hpp"

#include <cmath>
#include <vector>

using namespace zhilicon::kernels::dsp;

ZH_TEST(fir_identity) {
  // Single tap [1.0] is the identity filter.
  std::vector<float> taps = {1.0f};
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> out(input.size());
  std::size_t n = fir_apply<float>(taps.data(), 1, input.data(),
                                   input.size(), out.data());
  ZH_CHECK_EQ(n, input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    ZH_CHECK_NEAR(out[i], input[i], 1e-6);
  }
}

ZH_TEST(fir_two_tap_average) {
  // y[n] = 0.5 * x[n] + 0.5 * x[n-1] (two-point moving average).
  std::vector<float> taps = {0.5f, 0.5f};
  std::vector<float> input = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
  std::vector<float> out(4);
  std::size_t n = fir_apply<float>(taps.data(), taps.size(), input.data(),
                                   input.size(), out.data());
  ZH_CHECK_EQ(n, std::size_t{4});
  ZH_CHECK_NEAR(out[0], 3.0f, 1e-6);
  ZH_CHECK_NEAR(out[1], 5.0f, 1e-6);
  ZH_CHECK_NEAR(out[2], 7.0f, 1e-6);
  ZH_CHECK_NEAR(out[3], 9.0f, 1e-6);
}

ZH_TEST(fir_zero_pad_warmup) {
  // First tap_count - 1 outputs reflect zero padding.
  std::vector<float> taps = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<float> input = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> out(input.size());
  fir_apply_zero_pad<float>(taps.data(), taps.size(), input.data(),
                            input.size(), out.data());
  ZH_CHECK_NEAR(out[0], 0.25f, 1e-6);
  ZH_CHECK_NEAR(out[1], 0.5f, 1e-6);
  ZH_CHECK_NEAR(out[2], 0.75f, 1e-6);
  ZH_CHECK_NEAR(out[3], 1.0f, 1e-6);
  ZH_CHECK_NEAR(out[4], 1.0f, 1e-6);
}

ZH_TEST(fir_returns_zero_when_too_short) {
  std::vector<float> taps = {1.0f, 2.0f, 3.0f};
  std::vector<float> input = {1.0f, 2.0f};
  std::vector<float> out(1);
  std::size_t n = fir_apply<float>(taps.data(), taps.size(), input.data(),
                                   input.size(), out.data());
  ZH_CHECK_EQ(n, std::size_t{0});
}

ZH_TEST(fir_zero_taps_returns_zero) {
  std::vector<float> input = {1.0f, 2.0f};
  std::vector<float> out(2);
  std::size_t n = fir_apply<float>(nullptr, 0, input.data(), input.size(),
                                   out.data());
  ZH_CHECK_EQ(n, std::size_t{0});
}

ZH_TEST(fir_streaming_matches_zero_pad) {
  std::vector<float> taps = {0.1f, 0.2f, 0.3f, 0.4f};
  std::vector<float> input(64);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = std::sin(static_cast<float>(i) * 0.1f);
  }
  std::vector<float> ref(input.size());
  fir_apply_zero_pad<float>(taps.data(), taps.size(), input.data(),
                            input.size(), ref.data());

  StreamingFir<float> fir(taps.data(), taps.size());
  std::vector<float> stream(input.size());
  fir.process(input.data(), input.size(), stream.data());
  ZH_CHECK_EQ(fir.tap_count(), taps.size());

  for (std::size_t i = 0; i < input.size(); ++i) {
    ZH_CHECK_NEAR(stream[i], ref[i], 1e-6);
  }
}

ZH_TEST(fir_streaming_block_consistency) {
  // Splitting the input across two process() calls should produce the
  // same output as one call.
  std::vector<float> taps = {0.5f, -0.25f, 0.125f};
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<float> ref(input.size());
  StreamingFir<float> a(taps.data(), taps.size());
  a.process(input.data(), input.size(), ref.data());

  StreamingFir<float> b(taps.data(), taps.size());
  std::vector<float> got(input.size());
  b.process(input.data(), 4, got.data());
  b.process(input.data() + 4, input.size() - 4, got.data() + 4);
  for (std::size_t i = 0; i < input.size(); ++i) {
    ZH_CHECK_NEAR(got[i], ref[i], 1e-6);
  }
}

ZH_TEST(fir_streaming_reset) {
  std::vector<float> taps = {1.0f, 1.0f};
  StreamingFir<float> a(taps.data(), taps.size());
  std::vector<float> input = {1, 2, 3, 4};
  std::vector<float> first(input.size());
  std::vector<float> second(input.size());
  a.process(input.data(), input.size(), first.data());
  a.reset();
  a.process(input.data(), input.size(), second.data());
  for (std::size_t i = 0; i < input.size(); ++i) {
    ZH_CHECK_NEAR(first[i], second[i], 1e-6);
  }
}

ZH_TEST(direct_convolve_matches_fir) {
  // Convolution of an input with taps reversed should match FIR output.
  std::vector<float> a = {1, 2, 3, 4, 5};
  std::vector<float> b = {0.5f, 0.25f};
  std::vector<float> conv(a.size() + b.size() - 1);
  direct_convolve<float>(a.data(), a.size(), b.data(), b.size(), conv.data());
  // y[0] = 0.5, y[1] = 0.25 + 1.0, y[2] = 0.5 + 1.5, y[3] = 1.0 + 2.0,
  // y[4] = 1.5 + 2.5, y[5] = 2.5 (since 0.25*5 = 1.25 from a, 0)
  ZH_CHECK_NEAR(conv[0], 0.5f, 1e-6);
  ZH_CHECK_NEAR(conv[1], 1.25f, 1e-6);
  ZH_CHECK_NEAR(conv[2], 2.0f, 1e-6);
  ZH_CHECK_NEAR(conv[3], 2.75f, 1e-6);
  ZH_CHECK_NEAR(conv[4], 3.5f, 1e-6);
  ZH_CHECK_NEAR(conv[5], 1.25f, 1e-6);
}

ZH_TEST(fir_double_precision) {
  std::vector<double> taps = {0.5, 0.5};
  std::vector<double> input = {1.0, 3.0, 5.0, 7.0};
  std::vector<double> out(3);
  std::size_t n = fir_apply<double>(taps.data(), 2, input.data(), 4,
                                    out.data());
  ZH_CHECK_EQ(n, std::size_t{3});
  ZH_CHECK_NEAR(out[0], 2.0, 1e-12);
  ZH_CHECK_NEAR(out[1], 4.0, 1e-12);
  ZH_CHECK_NEAR(out[2], 6.0, 1e-12);
}

ZH_TEST(fir_streaming_set_taps_resets) {
  StreamingFir<float> a;
  std::vector<float> taps1 = {1.0f};
  a.set_taps(taps1.data(), taps1.size());
  ZH_CHECK_EQ(a.tap_count(), std::size_t{1});
  std::vector<float> taps2 = {0.5f, 0.5f, 0.5f};
  a.set_taps(taps2.data(), taps2.size());
  ZH_CHECK_EQ(a.tap_count(), std::size_t{3});
}

ZH_TEST_MAIN("dsp/fir")
