// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../conv/conv2d_naive.hpp"

#include <vector>

using namespace zhilicon::kernels::conv;

ZH_TEST(conv2d_dims_basic) {
  Conv2dParams p;
  p.H = 5;
  p.W = 5;
  p.Kh = 3;
  p.Kw = 3;
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(conv2d_output_dims(p, &Oh, &Ow));
  ZH_CHECK_EQ(Oh, std::size_t{3});
  ZH_CHECK_EQ(Ow, std::size_t{3});
}

ZH_TEST(conv2d_dims_with_padding_same) {
  // Padding 1 around a 5x5 with 3x3 kernel and stride 1 produces 5x5.
  Conv2dParams p;
  p.H = 5;
  p.W = 5;
  p.Kh = 3;
  p.Kw = 3;
  p.Ph = 1;
  p.Pw = 1;
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(conv2d_output_dims(p, &Oh, &Ow));
  ZH_CHECK_EQ(Oh, std::size_t{5});
  ZH_CHECK_EQ(Ow, std::size_t{5});
}

ZH_TEST(conv2d_identity_kernel_returns_input) {
  // 1x1 kernel of value 1.0 returns the input.
  Conv2dParams p;
  p.N = 1;
  p.Cin = 1;
  p.Cout = 1;
  p.H = 4;
  p.W = 4;
  p.Kh = 1;
  p.Kw = 1;
  std::vector<float> in(16);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
  std::vector<float> w = {1.0f};
  std::vector<float> out(16);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
  for (std::size_t i = 0; i < in.size(); ++i) {
    ZH_CHECK_NEAR(out[i], in[i], 0.0f);
  }
}

ZH_TEST(conv2d_known_3x3_average_kernel) {
  // 3x3 averaging kernel applied to a 3x3 image -> 1x1 output.
  Conv2dParams p;
  p.H = 3;
  p.W = 3;
  p.Kh = 3;
  p.Kw = 3;
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> w(9, 1.0f / 9.0f);
  std::vector<float> out(1);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
  // Sum 1..9 / 9 = 5.0
  ZH_CHECK_NEAR(out[0], 5.0f, 1e-5);
}

ZH_TEST(conv2d_with_bias) {
  Conv2dParams p;
  p.H = 2;
  p.W = 2;
  p.Kh = 1;
  p.Kw = 1;
  std::vector<float> in = {1, 2, 3, 4};
  std::vector<float> w = {2.0f};
  std::vector<float> b = {0.5f};
  std::vector<float> out(4);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), b.data(), p, out.data()));
  ZH_CHECK_NEAR(out[0], 2.5f, 1e-6);
  ZH_CHECK_NEAR(out[1], 4.5f, 1e-6);
  ZH_CHECK_NEAR(out[2], 6.5f, 1e-6);
  ZH_CHECK_NEAR(out[3], 8.5f, 1e-6);
}

ZH_TEST(conv2d_zero_padding_same_size_with_known_kernel) {
  // 3x3 cross kernel applied to a constant 1s map with same padding.
  Conv2dParams p;
  p.H = 3;
  p.W = 3;
  p.Kh = 3;
  p.Kw = 3;
  p.Ph = 1;
  p.Pw = 1;
  std::vector<float> in(9, 1.0f);
  std::vector<float> w(9, 0.0f);
  // Cross: centre + 4 neighbours.
  w[1] = 1.0f;  // top
  w[3] = 1.0f;  // left
  w[4] = 1.0f;  // centre
  w[5] = 1.0f;  // right
  w[7] = 1.0f;  // bottom
  std::vector<float> out(9);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
  // Top-left corner has 3 valid taps (centre + right + bottom).
  ZH_CHECK_NEAR(out[0], 3.0f, 1e-6);
  // Centre has all 5.
  ZH_CHECK_NEAR(out[4], 5.0f, 1e-6);
  // Bottom-right corner has 3.
  ZH_CHECK_NEAR(out[8], 3.0f, 1e-6);
}

ZH_TEST(conv2d_multi_channel_in) {
  Conv2dParams p;
  p.Cin = 2;
  p.H = 2;
  p.W = 2;
  p.Kh = 1;
  p.Kw = 1;
  std::vector<float> in = {1, 2, 3, 4, 10, 20, 30, 40};
  std::vector<float> w = {1.0f, 0.5f};
  std::vector<float> out(4);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
  // Each output position = ch0 * 1 + ch1 * 0.5.
  ZH_CHECK_NEAR(out[0], 1.0f + 5.0f, 1e-6);
  ZH_CHECK_NEAR(out[1], 2.0f + 10.0f, 1e-6);
  ZH_CHECK_NEAR(out[2], 3.0f + 15.0f, 1e-6);
  ZH_CHECK_NEAR(out[3], 4.0f + 20.0f, 1e-6);
}

ZH_TEST(conv2d_multi_channel_out) {
  Conv2dParams p;
  p.Cout = 2;
  p.H = 2;
  p.W = 2;
  p.Kh = 1;
  p.Kw = 1;
  std::vector<float> in = {1, 2, 3, 4};
  std::vector<float> w = {1.0f, 2.0f};
  std::vector<float> out(8);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
  // First 4: x * 1, second 4: x * 2.
  for (std::size_t i = 0; i < 4; ++i) {
    ZH_CHECK_NEAR(out[i], static_cast<float>(i + 1), 1e-6);
    ZH_CHECK_NEAR(out[4 + i], 2.0f * static_cast<float>(i + 1), 1e-6);
  }
}

ZH_TEST(conv2d_stride_2) {
  Conv2dParams p;
  p.H = 4;
  p.W = 4;
  p.Kh = 2;
  p.Kw = 2;
  p.Sh = 2;
  p.Sw = 2;
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8,
                            9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> w(4, 1.0f);
  std::vector<float> out(4);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
  // Sum of 4 cells in each 2x2 quadrant.
  ZH_CHECK_NEAR(out[0], 1.0f + 2.0f + 5.0f + 6.0f, 1e-6);
  ZH_CHECK_NEAR(out[1], 3.0f + 4.0f + 7.0f + 8.0f, 1e-6);
  ZH_CHECK_NEAR(out[2], 9.0f + 10.0f + 13.0f + 14.0f, 1e-6);
  ZH_CHECK_NEAR(out[3], 11.0f + 12.0f + 15.0f + 16.0f, 1e-6);
}

ZH_TEST(conv2d_zero_kernel_gives_bias) {
  Conv2dParams p;
  p.H = 3;
  p.W = 3;
  p.Kh = 3;
  p.Kw = 3;
  std::vector<float> in(9, 5.0f);
  std::vector<float> w(9, 0.0f);
  std::vector<float> b = {7.0f};
  std::vector<float> out(1);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), b.data(), p, out.data()));
  ZH_CHECK_NEAR(out[0], 7.0f, 1e-6);
}

ZH_TEST(conv2d_invalid_params_returns_false) {
  Conv2dParams p;
  p.Sh = 0;
  std::vector<float> in(1), w(1), out(1);
  ZH_CHECK(!conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
}

ZH_TEST(conv2d_kernel_too_large_for_input) {
  Conv2dParams p;
  p.H = 2;
  p.W = 2;
  p.Kh = 5;
  p.Kw = 5;
  std::vector<float> in(4), w(25), out(1);
  ZH_CHECK(!conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
}

ZH_TEST(conv2d_batch_n_2) {
  Conv2dParams p;
  p.N = 2;
  p.H = 2;
  p.W = 2;
  p.Kh = 1;
  p.Kw = 1;
  std::vector<float> in = {1, 2, 3, 4, 10, 20, 30, 40};
  std::vector<float> w = {3.0f};
  std::vector<float> out(8);
  ZH_CHECK(conv2d_naive<float>(in.data(), w.data(), nullptr, p, out.data()));
  for (std::size_t i = 0; i < 8; ++i) {
    ZH_CHECK_NEAR(out[i], 3.0f * in[i], 1e-6);
  }
}

ZH_TEST_MAIN("conv/conv2d")
