// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../quant/int8_mm.hpp"

#include <cstdint>
#include <random>
#include <vector>

using namespace zhilicon::kernels::quant;

ZH_TEST(int8_matmul_known_values) {
  // [1 -2; 3 4] x [-1 2; 0 3] = [-1 -4; -3 18]
  std::vector<std::int8_t> A = {1, -2, 3, 4};
  std::vector<std::int8_t> B = {-1, 2, 0, 3};
  std::vector<std::int32_t> C(4, 0);
  int8_matmul(A.data(), 2, B.data(), 2, C.data(), 2, 2, 2, 2);
  ZH_CHECK_EQ(C[0], -1);
  ZH_CHECK_EQ(C[1], -4);
  ZH_CHECK_EQ(C[2], -3);
  ZH_CHECK_EQ(C[3], 18);
}

ZH_TEST(int8_matmul_overflow_safe_extreme_values) {
  // -128 * -128 * 256 = 4194304, well within int32 range.
  // Build a 1x256 row of -128s and a 256x1 column of -128s.
  std::vector<std::int8_t> A(256, static_cast<std::int8_t>(-128));
  std::vector<std::int8_t> B(256, static_cast<std::int8_t>(-128));
  std::vector<std::int32_t> C(1, 0);
  int8_matmul(A.data(), 256, B.data(), 1, C.data(), 1, 1, 1, 256);
  ZH_CHECK_EQ(C[0], -128 * -128 * 256);
}

ZH_TEST(int8_matmul_max_dimension) {
  // Build a 1xK row of 127s and a Kx1 column of 127s; expected
  // accumulator = K * 127 * 127. We pick K = 1 << 15 so the result is
  // 127 * 127 * 32768 = 528514048, still below INT_MAX = 2147483647.
  const std::size_t K = 1u << 15;
  std::vector<std::int8_t> A(K, 127);
  std::vector<std::int8_t> B(K, 127);
  std::vector<std::int32_t> C(1, 0);
  int8_matmul(A.data(), K, B.data(), 1, C.data(), 1, 1, 1, K);
  std::int64_t expected = static_cast<std::int64_t>(127) * 127 *
                          static_cast<std::int64_t>(K);
  ZH_CHECK_EQ(static_cast<std::int64_t>(C[0]), expected);
}

ZH_TEST(int8_matmul_zero_inputs_give_zero) {
  std::vector<std::int8_t> A(64, 0);
  std::vector<std::int8_t> B(64, 0);
  std::vector<std::int32_t> C(1, 0);
  int8_matmul(A.data(), 64, B.data(), 1, C.data(), 1, 1, 1, 64);
  ZH_CHECK_EQ(C[0], 0);
}

ZH_TEST(int8_matmul_blocked_matches_naive) {
  const std::size_t M = 17, N = 19, K = 23;
  std::mt19937 rng(0xCAFE);
  std::uniform_int_distribution<int> dist(-128, 127);
  std::vector<std::int8_t> A(M * K), B(K * N);
  for (auto& v : A) v = static_cast<std::int8_t>(dist(rng));
  for (auto& v : B) v = static_cast<std::int8_t>(dist(rng));
  std::vector<std::int32_t> C_ref(M * N), C_blk(M * N);
  int8_matmul(A.data(), K, B.data(), N, C_ref.data(), N, M, N, K);
  int8_matmul_blocked<8>(A.data(), K, B.data(), N, C_blk.data(), N, M, N, K);
  for (std::size_t i = 0; i < C_ref.size(); ++i) {
    ZH_CHECK_EQ(C_blk[i], C_ref[i]);
  }
}

ZH_TEST(int8_matmul_blocked_16_matches_naive) {
  const std::size_t M = 32, N = 32, K = 32;
  std::mt19937 rng(0xBEEF);
  std::uniform_int_distribution<int> dist(-50, 50);
  std::vector<std::int8_t> A(M * K), B(K * N);
  for (auto& v : A) v = static_cast<std::int8_t>(dist(rng));
  for (auto& v : B) v = static_cast<std::int8_t>(dist(rng));
  std::vector<std::int32_t> C_ref(M * N), C_blk(M * N);
  int8_matmul(A.data(), K, B.data(), N, C_ref.data(), N, M, N, K);
  int8_matmul_blocked<16>(A.data(), K, B.data(), N, C_blk.data(), N, M, N, K);
  for (std::size_t i = 0; i < C_ref.size(); ++i) {
    ZH_CHECK_EQ(C_blk[i], C_ref[i]);
  }
}

ZH_TEST(int8_matvec_known_values) {
  std::vector<std::int8_t> A = {1, 2, 3, 4, 5, 6};   // 2x3
  std::vector<std::int8_t> x = {7, -1, 2};
  std::vector<std::int32_t> y(2, 0);
  int8_matvec(A.data(), 3, x.data(), y.data(), 2, 3);
  // y[0] = 1*7 + 2*-1 + 3*2 = 11
  // y[1] = 4*7 + 5*-1 + 6*2 = 35
  ZH_CHECK_EQ(y[0], 11);
  ZH_CHECK_EQ(y[1], 35);
}

ZH_TEST(int8_matvec_zero_input) {
  std::vector<std::int8_t> A(20, 7);
  std::vector<std::int8_t> x(5, 0);
  std::vector<std::int32_t> y(4, 99);
  int8_matvec(A.data(), 5, x.data(), y.data(), 4, 5);
  for (auto v : y) {
    ZH_CHECK_EQ(v, 0);
  }
}

ZH_TEST(int8_matmul_non_square) {
  std::vector<std::int8_t> A = {1, 2, 3};                      // 1x3
  std::vector<std::int8_t> B = {1, 0, 0, 0, 1, 0, 0, 0, 1};    // 3x3 identity
  std::vector<std::int32_t> C(3, 0);
  int8_matmul(A.data(), 3, B.data(), 3, C.data(), 3, 1, 3, 3);
  ZH_CHECK_EQ(C[0], 1);
  ZH_CHECK_EQ(C[1], 2);
  ZH_CHECK_EQ(C[2], 3);
}

ZH_TEST(int8_matmul_identity_b) {
  // A * I = A
  std::vector<std::int8_t> A = {-128, -64, 0, 63, 127};  // 1x5
  std::vector<std::int8_t> I(25, 0);
  for (std::size_t i = 0; i < 5; ++i) I[i * 5 + i] = 1;
  std::vector<std::int32_t> C(5, 0);
  int8_matmul(A.data(), 5, I.data(), 5, C.data(), 5, 1, 5, 5);
  ZH_CHECK_EQ(C[0], -128);
  ZH_CHECK_EQ(C[1], -64);
  ZH_CHECK_EQ(C[2], 0);
  ZH_CHECK_EQ(C[3], 63);
  ZH_CHECK_EQ(C[4], 127);
}

ZH_TEST_MAIN("quant/int8_mm")
