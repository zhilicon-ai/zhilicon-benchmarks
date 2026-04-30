// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../conv/im2col.hpp"

#include <vector>

using namespace zhilicon::kernels::conv;

ZH_TEST(im2col_dims_simple) {
  Im2colParams p;
  p.N = 1;
  p.C = 1;
  p.H = 4;
  p.W = 4;
  p.Kh = 3;
  p.Kw = 3;
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(im2col_output_dims(p, &Oh, &Ow));
  ZH_CHECK_EQ(Oh, std::size_t{2});
  ZH_CHECK_EQ(Ow, std::size_t{2});
}

ZH_TEST(im2col_dims_with_padding) {
  Im2colParams p;
  p.N = 1;
  p.C = 1;
  p.H = 4;
  p.W = 4;
  p.Kh = 3;
  p.Kw = 3;
  p.Ph = 1;
  p.Pw = 1;
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(im2col_output_dims(p, &Oh, &Ow));
  ZH_CHECK_EQ(Oh, std::size_t{4});
  ZH_CHECK_EQ(Ow, std::size_t{4});
}

ZH_TEST(im2col_rejects_zero_stride) {
  Im2colParams p;
  p.Sh = 0;
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(!im2col_output_dims(p, &Oh, &Ow));
}

ZH_TEST(im2col_basic_no_padding) {
  // 1x1x3x3 input, 2x2 kernel, no padding, stride 1.
  Im2colParams p;
  p.N = 1;
  p.C = 1;
  p.H = 3;
  p.W = 3;
  p.Kh = 2;
  p.Kw = 2;
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // Oh = Ow = 2, patch_rows = 4, spatial = 4.
  std::vector<float> out(16);
  ZH_CHECK(im2col(in.data(), p, out.data()));
  // First column patch (oh=0, ow=0): [1, 2, 4, 5]
  ZH_CHECK_NEAR(out[0 * 4 + 0], 1.0f, 0.0f);
  ZH_CHECK_NEAR(out[1 * 4 + 0], 2.0f, 0.0f);
  ZH_CHECK_NEAR(out[2 * 4 + 0], 4.0f, 0.0f);
  ZH_CHECK_NEAR(out[3 * 4 + 0], 5.0f, 0.0f);
  // Last patch (oh=1, ow=1): [5, 6, 8, 9]
  ZH_CHECK_NEAR(out[0 * 4 + 3], 5.0f, 0.0f);
  ZH_CHECK_NEAR(out[1 * 4 + 3], 6.0f, 0.0f);
  ZH_CHECK_NEAR(out[2 * 4 + 3], 8.0f, 0.0f);
  ZH_CHECK_NEAR(out[3 * 4 + 3], 9.0f, 0.0f);
}

ZH_TEST(im2col_with_padding_zero_pads_corner) {
  // 1x1x2x2 input padded by 1 -> 4x4 effective. 3x3 kernel, stride 1.
  Im2colParams p;
  p.H = 2;
  p.W = 2;
  p.Kh = 3;
  p.Kw = 3;
  p.Ph = 1;
  p.Pw = 1;
  std::vector<float> in = {1, 2, 3, 4};
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(im2col_output_dims(p, &Oh, &Ow));
  ZH_CHECK_EQ(Oh, std::size_t{2});
  ZH_CHECK_EQ(Ow, std::size_t{2});
  std::vector<float> out(9 * 4, 99.0f);
  ZH_CHECK(im2col(in.data(), p, out.data()));
  // Top-left patch corresponds to a 3x3 with the input at the bottom-right.
  // Top row (kh=0) should be all zero (padding).
  for (std::size_t kw = 0; kw < 3; ++kw) {
    ZH_CHECK_NEAR(out[(0 * 3 + kw) * 4 + 0], 0.0f, 0.0f);
  }
  // First column (kw=0) should be all zero (padding).
  for (std::size_t kh = 0; kh < 3; ++kh) {
    ZH_CHECK_NEAR(out[(kh * 3 + 0) * 4 + 0], 0.0f, 0.0f);
  }
  // Centre value of top-left patch is in[0,0] = 1.
  ZH_CHECK_NEAR(out[(1 * 3 + 1) * 4 + 0], 1.0f, 0.0f);
}

ZH_TEST(col2im_round_trips_unit_kernel) {
  // 1x1 kernel with stride 1: im2col is reshape, col2im is the reverse.
  Im2colParams p;
  p.H = 4;
  p.W = 5;
  p.Kh = 1;
  p.Kw = 1;
  std::vector<float> in(20);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
  std::vector<float> col(20);
  ZH_CHECK(im2col(in.data(), p, col.data()));
  std::vector<float> back(20, 0.0f);
  ZH_CHECK(col2im(col.data(), p, back.data()));
  for (std::size_t i = 0; i < in.size(); ++i) {
    ZH_CHECK_NEAR(back[i], in[i], 0.0f);
  }
}

ZH_TEST(im2col_stride_2) {
  Im2colParams p;
  p.H = 4;
  p.W = 4;
  p.Kh = 2;
  p.Kw = 2;
  p.Sh = 2;
  p.Sw = 2;
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(im2col_output_dims(p, &Oh, &Ow));
  ZH_CHECK_EQ(Oh, std::size_t{2});
  ZH_CHECK_EQ(Ow, std::size_t{2});
}

ZH_TEST(im2col_dilation_works) {
  Im2colParams p;
  p.H = 5;
  p.W = 5;
  p.Kh = 3;
  p.Kw = 3;
  p.Dh = 2;
  p.Dw = 2;
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(im2col_output_dims(p, &Oh, &Ow));
  // Effective kernel size is 1 + 2*(3-1) = 5, so output is 1x1.
  ZH_CHECK_EQ(Oh, std::size_t{1});
  ZH_CHECK_EQ(Ow, std::size_t{1});
}

ZH_TEST(im2col_kernel_too_large_returns_false) {
  Im2colParams p;
  p.H = 2;
  p.W = 2;
  p.Kh = 5;
  p.Kw = 5;
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(!im2col_output_dims(p, &Oh, &Ow));
}

ZH_TEST(im2col_multi_channel) {
  Im2colParams p;
  p.C = 2;
  p.H = 2;
  p.W = 2;
  p.Kh = 1;
  p.Kw = 1;
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8};  // 2 channels of 2x2
  std::size_t Oh = 0, Ow = 0;
  ZH_CHECK(im2col_output_dims(p, &Oh, &Ow));
  ZH_CHECK_EQ(Oh, std::size_t{2});
  ZH_CHECK_EQ(Ow, std::size_t{2});
  // patch_rows = C*Kh*Kw = 2, spatial = 4.
  std::vector<float> out(8);
  ZH_CHECK(im2col(in.data(), p, out.data()));
  for (std::size_t i = 0; i < 4; ++i) {
    ZH_CHECK_NEAR(out[0 * 4 + i], in[i], 0.0f);
    ZH_CHECK_NEAR(out[1 * 4 + i], in[4 + i], 0.0f);
  }
}

ZH_TEST(col2im_invalid_params_returns_false) {
  Im2colParams p;
  p.Sh = 0;
  std::vector<float> col(1), out(1);
  ZH_CHECK(!col2im(col.data(), p, out.data()));
}

ZH_TEST_MAIN("conv/im2col")
