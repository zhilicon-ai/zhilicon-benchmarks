// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: QR factorisation via Householder reflections.
//
// Given a real-valued M x N matrix A with M >= N, computes A = Q * R where
// Q is M x M orthogonal and R is M x N upper triangular. The factorisation
// is performed in-place on a working copy and the orthogonal factor is
// either accumulated explicitly into a M x M matrix or returned as a list
// of Householder vectors and scalars (compact form).
//
// References:
//   * Trefethen & Bau, "Numerical Linear Algebra", Lecture 10
//   * Golub & Van Loan, "Matrix Computations", Algorithm 5.2.1
//
// The implementation favours clarity over peak performance. It is intended
// to provide a numerical baseline against which the Zhilicon SDK QR kernels
// can be regression-tested.

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace zhilicon::kernels::linalg {

// Result of a Householder QR factorisation. R is stored in the upper
// triangle of `a` (the input matrix is overwritten); the strictly lower
// triangle holds the essential parts of the Householder vectors with
// their leading 1.0 implicit. tau contains the reflector scalars.
template <typename T>
struct QRResult {
  std::size_t M{0};
  std::size_t N{0};
  std::vector<T> a;     // packed R + Householder vectors, M x N row-major
  std::vector<T> tau;   // length min(M, N)
  bool ok{false};
};

namespace detail {

// Compute v and tau for a Householder reflector v such that
//     (I - tau * v * v^T) * x = ||x|| * e1
// The first element of v is normalised to 1 (implicit) but for the
// caller we store the explicit vector here for use in tests and the
// "form Q" routine.
template <typename T>
T householder_vector(const T* x, std::size_t n, T* v) noexcept {
  T sigma = T{0};
  for (std::size_t i = 1; i < n; ++i) {
    sigma += x[i] * x[i];
  }
  v[0] = T{1};
  for (std::size_t i = 1; i < n; ++i) {
    v[i] = x[i];
  }
  if (sigma == T{0} && x[0] >= T{0}) {
    return T{0};  // already a multiple of e1
  }
  T norm = std::sqrt(x[0] * x[0] + sigma);
  T v0;
  if (x[0] <= T{0}) {
    v0 = x[0] - norm;
  } else {
    // Avoid catastrophic cancellation: use -sigma / (x[0] + norm).
    v0 = -sigma / (x[0] + norm);
  }
  T tau = (T{2} * v0 * v0) / (sigma + v0 * v0);
  T inv = T{1} / v0;
  v[0] = T{1};
  for (std::size_t i = 1; i < n; ++i) {
    v[i] = x[i] * inv;
  }
  return tau;
}

}  // namespace detail

// Compute QR factorisation in-place. The input matrix `a_in` is M x N row-major.
// Returns a populated QRResult. If M < N, returns ok = false.
template <typename T>
QRResult<T> householder_qr(const T* a_in, std::size_t M, std::size_t N) {
  static_assert(std::is_floating_point<T>::value,
                "householder_qr requires a floating point T");
  QRResult<T> res;
  if (M == 0 || N == 0 || M < N) {
    return res;
  }
  res.M = M;
  res.N = N;
  res.a.assign(a_in, a_in + M * N);
  res.tau.assign(N, T{0});

  std::vector<T> v(M);
  std::vector<T> column(M);

  for (std::size_t k = 0; k < N; ++k) {
    const std::size_t len = M - k;
    // Extract column k from row k downward.
    for (std::size_t i = 0; i < len; ++i) {
      column[i] = res.a[(k + i) * N + k];
    }
    T tau = detail::householder_vector<T>(column.data(), len, v.data());
    res.tau[k] = tau;
    if (tau == T{0}) {
      // Already aligned; just store norm in-place (no reflection applied).
      continue;
    }
    // Update trailing submatrix A[k:, k:] = (I - tau v v^T) A[k:, k:].
    for (std::size_t j = k; j < N; ++j) {
      T dot = T{0};
      for (std::size_t i = 0; i < len; ++i) {
        dot += v[i] * res.a[(k + i) * N + j];
      }
      T scale = tau * dot;
      for (std::size_t i = 0; i < len; ++i) {
        res.a[(k + i) * N + j] -= scale * v[i];
      }
    }
    // Store the essential Householder vector below the diagonal in column k.
    for (std::size_t i = 1; i < len; ++i) {
      res.a[(k + i) * N + k] = v[i];
    }
  }
  res.ok = true;
  return res;
}

// Reconstruct the orthogonal factor Q (M x M) from a QRResult. The output
// buffer is row-major and must hold M*M elements.
template <typename T>
bool form_q(const QRResult<T>& qr, T* q_out) {
  if (!qr.ok) {
    return false;
  }
  const std::size_t M = qr.M;
  const std::size_t N = qr.N;
  // Initialise Q = I.
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      q_out[i * M + j] = (i == j) ? T{1} : T{0};
    }
  }
  std::vector<T> v(M);
  for (std::size_t k = N; k-- > 0;) {
    const std::size_t len = M - k;
    v[0] = T{1};
    for (std::size_t i = 1; i < len; ++i) {
      v[i] = qr.a[(k + i) * N + k];
    }
    T tau = qr.tau[k];
    if (tau == T{0}) {
      continue;
    }
    // Apply reflector to columns k..M-1 of Q from the left of the lower-
    // right submatrix Q[k:, k:].
    for (std::size_t j = k; j < M; ++j) {
      T dot = T{0};
      for (std::size_t i = 0; i < len; ++i) {
        dot += v[i] * q_out[(k + i) * M + j];
      }
      T scale = tau * dot;
      for (std::size_t i = 0; i < len; ++i) {
        q_out[(k + i) * M + j] -= scale * v[i];
      }
    }
  }
  return true;
}

// Extract the upper-triangular R factor from a QRResult into a separate
// M x N row-major buffer (zeroing the strictly-lower triangle).
template <typename T>
bool form_r(const QRResult<T>& qr, T* r_out) {
  if (!qr.ok) {
    return false;
  }
  const std::size_t M = qr.M;
  const std::size_t N = qr.N;
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      r_out[i * N + j] = (j >= i) ? qr.a[i * N + j] : T{0};
    }
  }
  return true;
}

}  // namespace zhilicon::kernels::linalg
