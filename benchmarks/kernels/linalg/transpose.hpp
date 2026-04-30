// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: cache-friendly matrix transpose.
//
// We expose three kernels:
//   * transpose_naive       - reference scalar implementation
//   * transpose_tiled       - cache-friendly blocked transpose with a
//                             compile-time tile size
//   * transpose_inplace_sq  - in-place square transpose (M == N), tiled

#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace zhilicon::kernels::linalg {

// Reference naive transpose: out[j, i] = in[i, j].
template <typename T>
void transpose_naive(const T* in, std::size_t in_ld, T* out,
                     std::size_t out_ld, std::size_t M, std::size_t N) noexcept {
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      out[j * out_ld + i] = in[i * in_ld + j];
    }
  }
}

// Tiled out-of-place transpose. Tile is a compile-time template parameter
// so the inner loops are unrolled. The output buffer must have enough
// rows for N (the post-transpose row count).
template <typename T, std::size_t Tile = 32>
void transpose_tiled(const T* in, std::size_t in_ld, T* out,
                     std::size_t out_ld, std::size_t M, std::size_t N) noexcept {
  static_assert(Tile > 0, "tile size must be positive");
  for (std::size_t ii = 0; ii < M; ii += Tile) {
    const std::size_t i_end = std::min(ii + Tile, M);
    for (std::size_t jj = 0; jj < N; jj += Tile) {
      const std::size_t j_end = std::min(jj + Tile, N);
      for (std::size_t i = ii; i < i_end; ++i) {
        for (std::size_t j = jj; j < j_end; ++j) {
          out[j * out_ld + i] = in[i * in_ld + j];
        }
      }
    }
  }
}

// In-place transpose for a square matrix. Returns true iff M == N (the
// only case supported in place).
template <typename T, std::size_t Tile = 16>
bool transpose_inplace_sq(T* a, std::size_t ld, std::size_t M,
                          std::size_t N) noexcept {
  if (M != N) {
    return false;
  }
  for (std::size_t ii = 0; ii < M; ii += Tile) {
    const std::size_t i_end = std::min(ii + Tile, M);
    for (std::size_t jj = ii; jj < N; jj += Tile) {
      const std::size_t j_end = std::min(jj + Tile, N);
      for (std::size_t i = ii; i < i_end; ++i) {
        const std::size_t j_start = (jj == ii) ? (i + 1) : jj;
        for (std::size_t j = j_start; j < j_end; ++j) {
          std::swap(a[i * ld + j], a[j * ld + i]);
        }
      }
    }
  }
  return true;
}

// Convenience packed-stride overloads.
template <typename T>
void transpose_naive_packed(const T* in, T* out, std::size_t M,
                            std::size_t N) noexcept {
  transpose_naive<T>(in, N, out, M, M, N);
}

template <typename T, std::size_t Tile = 32>
void transpose_tiled_packed(const T* in, T* out, std::size_t M,
                            std::size_t N) noexcept {
  transpose_tiled<T, Tile>(in, N, out, M, M, N);
}

}  // namespace zhilicon::kernels::linalg
