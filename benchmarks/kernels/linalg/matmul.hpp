// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: blocked / tiled matrix multiplication.
//
// Computes C = A * B for row-major matrices of compile-time tile size.
// We expose three kernels:
//   * matmul_naive   - reference triple loop, used by tests as the oracle
//   * matmul_blocked - cache-friendly blocked variant with explicit tile
//                      size template parameter (8, 16, 32 are exposed)
//   * matmul_blocked_alpha_beta - GEMM-style C = alpha * A * B + beta * C
//
// Dimensions are (M, K) x (K, N) -> (M, N). Strides are caller-provided
// so the caller can express slices of larger matrices without copying.

#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace zhilicon::kernels::linalg {

// Reference naive matmul. Used by tests as the ground truth.
template <typename T>
void matmul_naive(const T* A, std::size_t lda, const T* B, std::size_t ldb,
                  T* C, std::size_t ldc, std::size_t M, std::size_t N,
                  std::size_t K) noexcept {
  static_assert(std::is_arithmetic<T>::value, "matmul requires arithmetic T");
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T acc = T{0};
      for (std::size_t k = 0; k < K; ++k) {
        acc += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = acc;
    }
  }
}

// Blocked matmul. Tile is a compile-time template parameter so the inner
// loops are unrolled. We use the i-k-j loop order in the inner tile so
// that B is streamed sequentially.
template <typename T, std::size_t Tile>
void matmul_blocked(const T* A, std::size_t lda, const T* B, std::size_t ldb,
                    T* C, std::size_t ldc, std::size_t M, std::size_t N,
                    std::size_t K) noexcept {
  static_assert(std::is_arithmetic<T>::value, "matmul requires arithmetic T");
  static_assert(Tile > 0, "tile size must be positive");
  // Zero C first so accumulation is well-defined regardless of input
  // contents.
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      C[i * ldc + j] = T{0};
    }
  }
  for (std::size_t ii = 0; ii < M; ii += Tile) {
    const std::size_t i_end = std::min(ii + Tile, M);
    for (std::size_t kk = 0; kk < K; kk += Tile) {
      const std::size_t k_end = std::min(kk + Tile, K);
      for (std::size_t jj = 0; jj < N; jj += Tile) {
        const std::size_t j_end = std::min(jj + Tile, N);
        for (std::size_t i = ii; i < i_end; ++i) {
          for (std::size_t k = kk; k < k_end; ++k) {
            T a = A[i * lda + k];
            T* c_row = C + i * ldc;
            const T* b_row = B + k * ldb;
            for (std::size_t j = jj; j < j_end; ++j) {
              c_row[j] += a * b_row[j];
            }
          }
        }
      }
    }
  }
}

// GEMM-style alpha/beta variant: C = alpha * A * B + beta * C. Tile is
// chosen at compile time for the kernel; pick 16 for general use.
template <typename T, std::size_t Tile = 16>
void matmul_blocked_alpha_beta(T alpha, const T* A, std::size_t lda,
                               const T* B, std::size_t ldb, T beta, T* C,
                               std::size_t ldc, std::size_t M, std::size_t N,
                               std::size_t K) noexcept {
  static_assert(std::is_arithmetic<T>::value, "matmul requires arithmetic T");
  // Scale C by beta first.
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      C[i * ldc + j] *= beta;
    }
  }
  for (std::size_t ii = 0; ii < M; ii += Tile) {
    const std::size_t i_end = std::min(ii + Tile, M);
    for (std::size_t kk = 0; kk < K; kk += Tile) {
      const std::size_t k_end = std::min(kk + Tile, K);
      for (std::size_t jj = 0; jj < N; jj += Tile) {
        const std::size_t j_end = std::min(jj + Tile, N);
        for (std::size_t i = ii; i < i_end; ++i) {
          for (std::size_t k = kk; k < k_end; ++k) {
            T a = alpha * A[i * lda + k];
            T* c_row = C + i * ldc;
            const T* b_row = B + k * ldb;
            for (std::size_t j = jj; j < j_end; ++j) {
              c_row[j] += a * b_row[j];
            }
          }
        }
      }
    }
  }
}

// Convenience overloads for tightly packed (lda = K, ldb = N, ldc = N) cases.
template <typename T, std::size_t Tile>
void matmul_blocked_packed(const T* A, const T* B, T* C, std::size_t M,
                           std::size_t N, std::size_t K) noexcept {
  matmul_blocked<T, Tile>(A, K, B, N, C, N, M, N, K);
}

template <typename T>
void matmul_naive_packed(const T* A, const T* B, T* C, std::size_t M,
                         std::size_t N, std::size_t K) noexcept {
  matmul_naive<T>(A, K, B, N, C, N, M, N, K);
}

}  // namespace zhilicon::kernels::linalg
