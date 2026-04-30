// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: int8 x int8 -> int32 matrix multiplication.
//
// The Zhilicon SDK exposes int8 GEMM via vendor kernels. To keep the
// benchmark suite hardware-agnostic we provide a portable scalar reference
// implementation that uses int32 accumulators, suitable as a numerical
// oracle. The kernel is written in plain C++17 so the compiler can fold
// it into vectorised assembly automatically when possible.
//
// Layout: row-major. A is (M x K) int8, B is (K x N) int8, C is (M x N)
// int32. The result is the exact integer dot product; no scale or zero
// point handling is performed at this layer (those live in the quantize
// header so the two stages can be benchmarked independently).

#pragma once

#include <cstddef>
#include <cstdint>

namespace zhilicon::kernels::quant {

// Reference int8 GEMM with int32 accumulator.
inline void int8_matmul(const std::int8_t* A, std::size_t lda,
                        const std::int8_t* B, std::size_t ldb,
                        std::int32_t* C, std::size_t ldc, std::size_t M,
                        std::size_t N, std::size_t K) noexcept {
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      std::int32_t acc = 0;
      for (std::size_t k = 0; k < K; ++k) {
        // Promote operands to int32 explicitly so we never sign-extend a
        // narrower intermediate.
        std::int32_t a = static_cast<std::int32_t>(A[i * lda + k]);
        std::int32_t b = static_cast<std::int32_t>(B[k * ldb + j]);
        acc += a * b;
      }
      C[i * ldc + j] = acc;
    }
  }
}

// Blocked variant for cache locality. Tile is a compile-time parameter.
template <std::size_t Tile = 16>
inline void int8_matmul_blocked(const std::int8_t* A, std::size_t lda,
                                const std::int8_t* B, std::size_t ldb,
                                std::int32_t* C, std::size_t ldc,
                                std::size_t M, std::size_t N,
                                std::size_t K) noexcept {
  static_assert(Tile > 0, "tile size must be positive");
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      C[i * ldc + j] = 0;
    }
  }
  for (std::size_t ii = 0; ii < M; ii += Tile) {
    std::size_t i_end = ii + Tile;
    if (i_end > M) i_end = M;
    for (std::size_t kk = 0; kk < K; kk += Tile) {
      std::size_t k_end = kk + Tile;
      if (k_end > K) k_end = K;
      for (std::size_t jj = 0; jj < N; jj += Tile) {
        std::size_t j_end = jj + Tile;
        if (j_end > N) j_end = N;
        for (std::size_t i = ii; i < i_end; ++i) {
          for (std::size_t k = kk; k < k_end; ++k) {
            std::int32_t a = static_cast<std::int32_t>(A[i * lda + k]);
            std::int32_t* c_row = C + i * ldc;
            const std::int8_t* b_row = B + k * ldb;
            for (std::size_t j = jj; j < j_end; ++j) {
              c_row[j] += a * static_cast<std::int32_t>(b_row[j]);
            }
          }
        }
      }
    }
  }
}

// Per-row dot product: out[i] = sum_k A[i, k] * B[k]. This is a building
// block for low-rank quantized projection layers.
inline void int8_matvec(const std::int8_t* A, std::size_t lda,
                        const std::int8_t* x, std::int32_t* y,
                        std::size_t M, std::size_t K) noexcept {
  for (std::size_t i = 0; i < M; ++i) {
    std::int32_t acc = 0;
    for (std::size_t k = 0; k < K; ++k) {
      acc += static_cast<std::int32_t>(A[i * lda + k])
             * static_cast<std::int32_t>(x[k]);
    }
    y[i] = acc;
  }
}

}  // namespace zhilicon::kernels::quant
