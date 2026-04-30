// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../linalg/matmul.hpp"

#include <cstdint>
#include <random>
#include <vector>

using namespace zhilicon::kernels::linalg;

namespace {

template <typename T>
void random_matrix(std::vector<T>& m, std::size_t rows, std::size_t cols,
                   std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  m.resize(rows * cols);
  for (auto& x : m) {
    x = static_cast<T>(dist(rng));
  }
}

template <typename T>
double max_diff(const std::vector<T>& a, const std::vector<T>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    if (d < 0) d = -d;
    if (d > m) m = d;
  }
  return m;
}

}  // namespace

ZH_TEST(matmul_naive_known_values) {
  // [1 2; 3 4] x [5 6; 7 8] = [19 22; 43 50]
  std::vector<float> A = {1, 2, 3, 4};
  std::vector<float> B = {5, 6, 7, 8};
  std::vector<float> C(4, 0.0f);
  matmul_naive_packed<float>(A.data(), B.data(), C.data(), 2, 2, 2);
  ZH_CHECK_NEAR(C[0], 19.0f, 1e-6);
  ZH_CHECK_NEAR(C[1], 22.0f, 1e-6);
  ZH_CHECK_NEAR(C[2], 43.0f, 1e-6);
  ZH_CHECK_NEAR(C[3], 50.0f, 1e-6);
}

ZH_TEST(matmul_blocked_8_matches_naive_floats) {
  const std::size_t M = 17, N = 19, K = 23;  // non-multiples of tile
  std::vector<float> A, B;
  random_matrix(A, M, K, 42);
  random_matrix(B, K, N, 24);
  std::vector<float> C_ref(M * N), C_blk(M * N);
  matmul_naive_packed<float>(A.data(), B.data(), C_ref.data(), M, N, K);
  matmul_blocked_packed<float, 8>(A.data(), B.data(), C_blk.data(), M, N, K);
  ZH_CHECK(max_diff(C_ref, C_blk) < 1e-3);
}

ZH_TEST(matmul_blocked_16_matches_naive_floats) {
  const std::size_t M = 32, N = 32, K = 32;
  std::vector<float> A, B;
  random_matrix(A, M, K, 1);
  random_matrix(B, K, N, 2);
  std::vector<float> C_ref(M * N), C_blk(M * N);
  matmul_naive_packed<float>(A.data(), B.data(), C_ref.data(), M, N, K);
  matmul_blocked_packed<float, 16>(A.data(), B.data(), C_blk.data(), M, N, K);
  ZH_CHECK(max_diff(C_ref, C_blk) < 1e-3);
}

ZH_TEST(matmul_blocked_32_matches_naive_doubles) {
  const std::size_t M = 64, N = 48, K = 40;
  std::vector<double> A, B;
  random_matrix(A, M, K, 7);
  random_matrix(B, K, N, 8);
  std::vector<double> C_ref(M * N), C_blk(M * N);
  matmul_naive_packed<double>(A.data(), B.data(), C_ref.data(), M, N, K);
  matmul_blocked_packed<double, 32>(A.data(), B.data(), C_blk.data(),
                                    M, N, K);
  ZH_CHECK(max_diff(C_ref, C_blk) < 1e-9);
}

ZH_TEST(matmul_identity_against_random) {
  const std::size_t N = 12;
  std::vector<double> I(N * N, 0.0);
  for (std::size_t i = 0; i < N; ++i) I[i * N + i] = 1.0;
  std::vector<double> X;
  random_matrix(X, N, N, 9);
  std::vector<double> result(N * N);
  matmul_blocked_packed<double, 16>(I.data(), X.data(), result.data(),
                                    N, N, N);
  ZH_CHECK(max_diff(result, X) < 1e-12);
}

ZH_TEST(matmul_alpha_beta_zero_beta) {
  // alpha = 1, beta = 0 -> regular matmul.
  const std::size_t M = 4, N = 4, K = 4;
  std::vector<float> A, B;
  random_matrix(A, M, K, 11);
  random_matrix(B, K, N, 12);
  std::vector<float> C_ref(M * N);
  std::vector<float> C(M * N, 99.0f);
  matmul_naive_packed<float>(A.data(), B.data(), C_ref.data(), M, N, K);
  matmul_blocked_alpha_beta<float, 16>(1.0f, A.data(), K, B.data(), N, 0.0f,
                                       C.data(), N, M, N, K);
  ZH_CHECK(max_diff(C_ref, C) < 1e-4);
}

ZH_TEST(matmul_alpha_beta_nonzero_beta) {
  // C := 2 * A * B + 0.5 * C, with C initialised to 1.0.
  const std::size_t M = 4, N = 4, K = 4;
  std::vector<float> A, B;
  random_matrix(A, M, K, 13);
  random_matrix(B, K, N, 14);
  std::vector<float> C_ref(M * N);
  matmul_naive_packed<float>(A.data(), B.data(), C_ref.data(), M, N, K);
  for (auto& v : C_ref) v *= 2.0f;
  for (auto& v : C_ref) v += 0.5f;

  std::vector<float> C(M * N, 1.0f);
  matmul_blocked_alpha_beta<float, 16>(2.0f, A.data(), K, B.data(), N, 0.5f,
                                       C.data(), N, M, N, K);
  ZH_CHECK(max_diff(C_ref, C) < 1e-4);
}

ZH_TEST(matmul_zero_dimensions_do_no_writes) {
  // M = 0, K = 0, N = 0 should be a no-op.
  std::vector<float> A, B, C;
  matmul_blocked_packed<float, 16>(A.data(), B.data(), C.data(), 0, 0, 0);
  matmul_naive_packed<float>(A.data(), B.data(), C.data(), 0, 0, 0);
  ZH_CHECK_EQ(C.size(), std::size_t{0});
}

ZH_TEST(matmul_single_element) {
  std::vector<double> A = {3.0};
  std::vector<double> B = {4.0};
  std::vector<double> C(1, 0.0);
  matmul_blocked_packed<double, 8>(A.data(), B.data(), C.data(), 1, 1, 1);
  ZH_CHECK_NEAR(C[0], 12.0, 1e-12);
}

ZH_TEST(matmul_tall_skinny) {
  const std::size_t M = 100, N = 1, K = 4;
  std::vector<double> A, B;
  random_matrix(A, M, K, 33);
  random_matrix(B, K, N, 34);
  std::vector<double> C_ref(M * N), C_blk(M * N);
  matmul_naive_packed<double>(A.data(), B.data(), C_ref.data(), M, N, K);
  matmul_blocked_packed<double, 16>(A.data(), B.data(), C_blk.data(),
                                    M, N, K);
  ZH_CHECK(max_diff(C_ref, C_blk) < 1e-9);
}

ZH_TEST(matmul_short_fat) {
  const std::size_t M = 1, N = 100, K = 4;
  std::vector<double> A, B;
  random_matrix(A, M, K, 35);
  random_matrix(B, K, N, 36);
  std::vector<double> C_ref(M * N), C_blk(M * N);
  matmul_naive_packed<double>(A.data(), B.data(), C_ref.data(), M, N, K);
  matmul_blocked_packed<double, 16>(A.data(), B.data(), C_blk.data(),
                                    M, N, K);
  ZH_CHECK(max_diff(C_ref, C_blk) < 1e-9);
}

ZH_TEST_MAIN("linalg/matmul")
