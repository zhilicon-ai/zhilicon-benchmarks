// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../linalg/qr_householder.hpp"
#include "../linalg/matmul.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace zhilicon::kernels::linalg;

namespace {

template <typename T>
double max_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > m) m = d;
  }
  return m;
}

template <typename T>
std::vector<T> q_times_r(const std::vector<T>& Q, std::size_t M,
                         const std::vector<T>& R, std::size_t N) {
  std::vector<T> out(M * N);
  matmul_naive_packed<T>(Q.data(), R.data(), out.data(), M, N, M);
  return out;
}

}  // namespace

ZH_TEST(qr_identity_matrix) {
  const std::size_t M = 4, N = 4;
  std::vector<double> A(M * N, 0.0);
  for (std::size_t i = 0; i < M; ++i) A[i * N + i] = 1.0;
  auto qr = householder_qr<double>(A.data(), M, N);
  ZH_CHECK(qr.ok);
  std::vector<double> Q(M * M), R(M * N);
  ZH_CHECK(form_q(qr, Q.data()));
  ZH_CHECK(form_r(qr, R.data()));
  // Q must be the identity, R must be the identity.
  std::vector<double> I(M * M, 0.0);
  for (std::size_t i = 0; i < M; ++i) I[i * M + i] = 1.0;
  ZH_CHECK(max_abs_diff(Q, I) < 1e-12);
  ZH_CHECK(max_abs_diff(R, A) < 1e-12);
}

ZH_TEST(qr_small_3x3_reconstructs_a) {
  const std::size_t M = 3, N = 3;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0};
  auto qr = householder_qr<double>(A.data(), M, N);
  ZH_CHECK(qr.ok);
  std::vector<double> Q(M * M), R(M * N);
  form_q(qr, Q.data());
  form_r(qr, R.data());
  auto qr_product = q_times_r(Q, M, R, N);
  ZH_CHECK(max_abs_diff(qr_product, A) < 1e-9);
}

ZH_TEST(qr_random_square_reconstructs_a) {
  const std::size_t M = 8, N = 8;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::vector<double> A(M * N);
  for (auto& v : A) v = dist(rng);
  auto qr = householder_qr<double>(A.data(), M, N);
  ZH_CHECK(qr.ok);
  std::vector<double> Q(M * M), R(M * N);
  form_q(qr, Q.data());
  form_r(qr, R.data());
  auto qr_product = q_times_r(Q, M, R, N);
  ZH_CHECK(max_abs_diff(qr_product, A) < 1e-8);
}

ZH_TEST(qr_random_tall_reconstructs_a) {
  const std::size_t M = 12, N = 5;
  std::mt19937 rng(99);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> A(M * N);
  for (auto& v : A) v = dist(rng);
  auto qr = householder_qr<double>(A.data(), M, N);
  ZH_CHECK(qr.ok);
  std::vector<double> Q(M * M), R(M * N);
  form_q(qr, Q.data());
  form_r(qr, R.data());
  auto qr_product = q_times_r(Q, M, R, N);
  ZH_CHECK(max_abs_diff(qr_product, A) < 1e-8);
}

ZH_TEST(qr_q_is_orthogonal) {
  const std::size_t M = 6, N = 6;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> A(M * N);
  for (auto& v : A) v = dist(rng);
  auto qr = householder_qr<double>(A.data(), M, N);
  ZH_CHECK(qr.ok);
  std::vector<double> Q(M * M);
  form_q(qr, Q.data());
  // Compute Q^T * Q via two transposes of Q; here we use a hand-rolled
  // dot product for simplicity.
  std::vector<double> QtQ(M * M, 0.0);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < M; ++k) {
        s += Q[k * M + i] * Q[k * M + j];
      }
      QtQ[i * M + j] = s;
    }
  }
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      double expected = (i == j) ? 1.0 : 0.0;
      ZH_CHECK_NEAR(QtQ[i * M + j], expected, 1e-9);
    }
  }
}

ZH_TEST(qr_r_is_upper_triangular) {
  const std::size_t M = 5, N = 4;
  std::mt19937 rng(13);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::vector<double> A(M * N);
  for (auto& v : A) v = dist(rng);
  auto qr = householder_qr<double>(A.data(), M, N);
  ZH_CHECK(qr.ok);
  std::vector<double> R(M * N);
  form_r(qr, R.data());
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (j < i) {
        ZH_CHECK_NEAR(R[i * N + j], 0.0, 1e-12);
      }
    }
  }
}

ZH_TEST(qr_rejects_short_fat) {
  std::vector<double> A(6);
  auto qr = householder_qr<double>(A.data(), 2, 3);
  ZH_CHECK(!qr.ok);
}

ZH_TEST(qr_zero_dimensions) {
  std::vector<double> A;
  auto qr = householder_qr<double>(A.data(), 0, 0);
  ZH_CHECK(!qr.ok);
}

ZH_TEST(qr_form_q_rejects_failed_result) {
  QRResult<double> bad;
  bad.ok = false;
  std::vector<double> Q(16);
  ZH_CHECK(!form_q(bad, Q.data()));
  ZH_CHECK(!form_r(bad, Q.data()));
}

ZH_TEST(qr_diagonal_input) {
  const std::size_t M = 4, N = 4;
  std::vector<double> A(M * N, 0.0);
  for (std::size_t i = 0; i < M; ++i) A[i * N + i] = static_cast<double>(i + 1);
  auto qr = householder_qr<double>(A.data(), M, N);
  ZH_CHECK(qr.ok);
  std::vector<double> Q(M * M), R(M * N);
  form_q(qr, Q.data());
  form_r(qr, R.data());
  auto qr_product = q_times_r(Q, M, R, N);
  ZH_CHECK(max_abs_diff(qr_product, A) < 1e-12);
}

ZH_TEST_MAIN("linalg/qr")
