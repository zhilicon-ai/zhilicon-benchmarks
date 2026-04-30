// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: dequantize-then-matmul fused reference.
//
// For benchmark purposes we provide a reference fused kernel that takes
// int8 inputs along with their per-tensor quantization parameters and
// returns the float matmul result. This composes int8_matmul and
// dequantize, but the fused form lets benchmarks measure end-to-end
// quantized-matmul cost without needing two separate buffers.

#pragma once

#include <cstddef>
#include <cstdint>

#include "int8_mm.hpp"
#include "quantize.hpp"

namespace zhilicon::kernels::quant {

// Compute C_float = (A_int8 - zpA) * scaleA * (B_int8 - zpB) * scaleB.
// Both operands are assumed to share the same per-tensor quantization
// schema; per-channel scaling lives in another header.
inline void dequant_matmul(const std::int8_t* A, std::size_t lda,
                           const QuantParams& pA, const std::int8_t* B,
                           std::size_t ldb, const QuantParams& pB,
                           float* C, std::size_t ldc, std::size_t M,
                           std::size_t N, std::size_t K) noexcept {
  // Compute the int32 result first.
  // Use a temporary stack-friendly buffer when M*N is small; otherwise we
  // assume the caller supplies workspace via ldc-strided rows.
  // For simplicity we accumulate directly into floats, which costs an
  // extra cast per inner iteration but avoids a large transient buffer.
  const float combined_scale = pA.scale * pB.scale;
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      std::int32_t acc = 0;
      for (std::size_t k = 0; k < K; ++k) {
        std::int32_t a = static_cast<std::int32_t>(A[i * lda + k]) -
                         pA.zero_point;
        std::int32_t b = static_cast<std::int32_t>(B[k * ldb + j]) -
                         pB.zero_point;
        acc += a * b;
      }
      C[i * ldc + j] = static_cast<float>(acc) * combined_scale;
    }
  }
}

// Per-row scaling variant used by transformer attention quantization
// schemes (e.g. SmoothQuant). row_scales has length M, col_scales has
// length N.
inline void dequant_matmul_per_row_col(const std::int8_t* A, std::size_t lda,
                                       const float* row_scales,
                                       const std::int8_t* B,
                                       std::size_t ldb,
                                       const float* col_scales, float* C,
                                       std::size_t ldc, std::size_t M,
                                       std::size_t N,
                                       std::size_t K) noexcept {
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      std::int32_t acc = 0;
      for (std::size_t k = 0; k < K; ++k) {
        acc += static_cast<std::int32_t>(A[i * lda + k]) *
               static_cast<std::int32_t>(B[k * ldb + j]);
      }
      C[i * ldc + j] = static_cast<float>(acc) * row_scales[i] *
                       col_scales[j];
    }
  }
}

}  // namespace zhilicon::kernels::quant
