// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../linalg/transpose.hpp"

#include <cstdint>
#include <random>
#include <vector>

using namespace zhilicon::kernels::linalg;

ZH_TEST(transpose_naive_2x3) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6};
  std::vector<int> out(6);
  transpose_naive_packed<int>(in.data(), out.data(), 2, 3);
  // Original 2x3 row-major:
  //   [1 2 3]
  //   [4 5 6]
  // Transposed 3x2:
  //   [1 4]
  //   [2 5]
  //   [3 6]
  ZH_CHECK_EQ(out[0], 1);
  ZH_CHECK_EQ(out[1], 4);
  ZH_CHECK_EQ(out[2], 2);
  ZH_CHECK_EQ(out[3], 5);
  ZH_CHECK_EQ(out[4], 3);
  ZH_CHECK_EQ(out[5], 6);
}

ZH_TEST(transpose_tiled_matches_naive_floats) {
  const std::size_t M = 37, N = 41;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> in(M * N);
  for (auto& v : in) v = dist(rng);
  std::vector<float> ref(N * M);
  transpose_naive_packed<float>(in.data(), ref.data(), M, N);
  std::vector<float> got(N * M);
  transpose_tiled_packed<float, 16>(in.data(), got.data(), M, N);
  for (std::size_t i = 0; i < ref.size(); ++i) {
    ZH_CHECK_EQ(ref[i], got[i]);
  }
}

ZH_TEST(transpose_tiled_matches_naive_doubles) {
  const std::size_t M = 64, N = 32;
  std::mt19937 rng(456);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> in(M * N);
  for (auto& v : in) v = dist(rng);
  std::vector<double> ref(N * M);
  transpose_naive_packed<double>(in.data(), ref.data(), M, N);
  std::vector<double> got(N * M);
  transpose_tiled_packed<double, 32>(in.data(), got.data(), M, N);
  for (std::size_t i = 0; i < ref.size(); ++i) {
    ZH_CHECK_EQ(ref[i], got[i]);
  }
}

ZH_TEST(transpose_inplace_square_matches_out_of_place) {
  const std::size_t N = 16;
  std::mt19937 rng(789);
  std::uniform_int_distribution<int> dist(-1000, 1000);
  std::vector<int> a(N * N);
  for (auto& v : a) v = dist(rng);
  std::vector<int> ref(N * N);
  transpose_naive_packed<int>(a.data(), ref.data(), N, N);
  std::vector<int> in_place = a;
  bool ok = transpose_inplace_sq<int, 8>(in_place.data(), N, N, N);
  ZH_CHECK(ok);
  for (std::size_t i = 0; i < ref.size(); ++i) {
    ZH_CHECK_EQ(in_place[i], ref[i]);
  }
}

ZH_TEST(transpose_inplace_rejects_non_square) {
  std::vector<int> a(6);
  bool ok = transpose_inplace_sq<int, 8>(a.data(), 3, 2, 3);
  ZH_CHECK(!ok);
}

ZH_TEST(transpose_involution_property) {
  const std::size_t M = 7, N = 11;
  std::mt19937 rng(111);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> a(M * N);
  for (auto& v : a) v = dist(rng);
  std::vector<float> at(N * M);
  std::vector<float> att(M * N);
  transpose_tiled_packed<float, 16>(a.data(), at.data(), M, N);
  transpose_tiled_packed<float, 16>(at.data(), att.data(), N, M);
  for (std::size_t i = 0; i < a.size(); ++i) {
    ZH_CHECK_EQ(att[i], a[i]);
  }
}

ZH_TEST(transpose_with_strides) {
  // Transpose with arbitrary strides (not equal to row size).
  // 4x3 input embedded in a 4x5 storage, transposed into a 3x4 output
  // embedded in a 3x6 storage.
  const std::size_t M = 4, N = 3;
  const std::size_t in_ld = 5, out_ld = 6;
  std::vector<int> in(M * in_ld, 0);
  std::vector<int> out(N * out_ld, 0);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      in[i * in_ld + j] = static_cast<int>(i * 10 + j);
    }
  }
  transpose_tiled<int, 8>(in.data(), in_ld, out.data(), out_ld, M, N);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      ZH_CHECK_EQ(out[j * out_ld + i], static_cast<int>(i * 10 + j));
    }
  }
}

ZH_TEST(transpose_single_row) {
  std::vector<int> in = {1, 2, 3, 4};
  std::vector<int> out(4);
  transpose_tiled_packed<int, 8>(in.data(), out.data(), 1, 4);
  for (std::size_t i = 0; i < 4; ++i) {
    ZH_CHECK_EQ(out[i], in[i]);
  }
}

ZH_TEST(transpose_single_column) {
  std::vector<int> in = {1, 2, 3, 4};
  std::vector<int> out(4);
  transpose_tiled_packed<int, 8>(in.data(), out.data(), 4, 1);
  for (std::size_t i = 0; i < 4; ++i) {
    ZH_CHECK_EQ(out[i], in[i]);
  }
}

ZH_TEST_MAIN("linalg/transpose")
