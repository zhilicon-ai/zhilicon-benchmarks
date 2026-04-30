// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: im2col reshape kernel.
//
// Converts a 4-D NCHW tensor into a 2-D (C*Kh*Kw) x (N*Oh*Ow) buffer so
// that a matmul can express the convolution. The layout matches the
// classical Caffe convention.
//
// Parameters:
//   * input:    pointer to N x C x H x W float buffer (row-major NCHW)
//   * N, C, H, W: input dimensions
//   * Kh, Kw:   kernel spatial dimensions
//   * Sh, Sw:   stride
//   * Ph, Pw:   zero-padding
//   * Dh, Dw:   dilation (1 = no dilation)
//   * output:   pointer to a buffer of size N * (C*Kh*Kw) * Oh * Ow
//
// Output spatial dimensions:
//   Oh = (H + 2*Ph - Dh*(Kh-1) - 1) / Sh + 1
//   Ow = (W + 2*Pw - Dw*(Kw-1) - 1) / Sw + 1
//
// Boundary handling: out-of-bounds reads produced by padding are treated
// as zero. This is the convention assumed by all downstream conv tests.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace zhilicon::kernels::conv {

struct Im2colParams {
  std::size_t N{1};
  std::size_t C{1};
  std::size_t H{1};
  std::size_t W{1};
  std::size_t Kh{1};
  std::size_t Kw{1};
  std::size_t Sh{1};
  std::size_t Sw{1};
  std::size_t Ph{0};
  std::size_t Pw{0};
  std::size_t Dh{1};
  std::size_t Dw{1};
};

// Compute the output spatial dimensions for an im2col operation. Returns
// false if the configuration is degenerate (e.g. zero stride or kernel
// extending past the image even with padding).
inline bool im2col_output_dims(const Im2colParams& p, std::size_t* Oh,
                               std::size_t* Ow) noexcept {
  if (p.Sh == 0 || p.Sw == 0 || p.Kh == 0 || p.Kw == 0 || p.Dh == 0 ||
      p.Dw == 0) {
    return false;
  }
  std::ptrdiff_t kh_eff = static_cast<std::ptrdiff_t>(p.Dh) *
                          (static_cast<std::ptrdiff_t>(p.Kh) - 1) + 1;
  std::ptrdiff_t kw_eff = static_cast<std::ptrdiff_t>(p.Dw) *
                          (static_cast<std::ptrdiff_t>(p.Kw) - 1) + 1;
  std::ptrdiff_t h_avail = static_cast<std::ptrdiff_t>(p.H) +
                           2 * static_cast<std::ptrdiff_t>(p.Ph) - kh_eff;
  std::ptrdiff_t w_avail = static_cast<std::ptrdiff_t>(p.W) +
                           2 * static_cast<std::ptrdiff_t>(p.Pw) - kw_eff;
  if (h_avail < 0 || w_avail < 0) {
    return false;
  }
  *Oh = static_cast<std::size_t>(h_avail) / p.Sh + 1;
  *Ow = static_cast<std::size_t>(w_avail) / p.Sw + 1;
  return true;
}

// Apply im2col. The output buffer has shape (N, C*Kh*Kw, Oh*Ow) but we
// flatten to a single contiguous array: out[n * patch_rows * spatial +
// row * spatial + col] where patch_rows = C*Kh*Kw and spatial = Oh*Ow.
template <typename T>
bool im2col(const T* input, const Im2colParams& p, T* output) noexcept {
  static_assert(std::is_arithmetic<T>::value, "im2col requires arithmetic T");
  std::size_t Oh = 0;
  std::size_t Ow = 0;
  if (!im2col_output_dims(p, &Oh, &Ow)) {
    return false;
  }
  const std::size_t patch_rows = p.C * p.Kh * p.Kw;
  const std::size_t spatial = Oh * Ow;
  for (std::size_t n = 0; n < p.N; ++n) {
    const T* in_n = input + n * p.C * p.H * p.W;
    T* out_n = output + n * patch_rows * spatial;
    for (std::size_t c = 0; c < p.C; ++c) {
      for (std::size_t kh = 0; kh < p.Kh; ++kh) {
        for (std::size_t kw = 0; kw < p.Kw; ++kw) {
          std::size_t row = c * p.Kh * p.Kw + kh * p.Kw + kw;
          T* out_row = out_n + row * spatial;
          for (std::size_t oh = 0; oh < Oh; ++oh) {
            for (std::size_t ow = 0; ow < Ow; ++ow) {
              std::ptrdiff_t in_h = static_cast<std::ptrdiff_t>(oh * p.Sh +
                                    kh * p.Dh) -
                                    static_cast<std::ptrdiff_t>(p.Ph);
              std::ptrdiff_t in_w = static_cast<std::ptrdiff_t>(ow * p.Sw +
                                    kw * p.Dw) -
                                    static_cast<std::ptrdiff_t>(p.Pw);
              if (in_h < 0 || in_h >= static_cast<std::ptrdiff_t>(p.H) ||
                  in_w < 0 || in_w >= static_cast<std::ptrdiff_t>(p.W)) {
                out_row[oh * Ow + ow] = T{0};
              } else {
                out_row[oh * Ow + ow] = in_n[c * p.H * p.W +
                                             static_cast<std::size_t>(in_h) *
                                                 p.W +
                                             static_cast<std::size_t>(in_w)];
              }
            }
          }
        }
      }
    }
  }
  return true;
}

// Inverse: col2im. Accumulates patches back into the input layout. Used
// by the backward pass of convolution. Out-of-bounds positions are
// silently dropped.
template <typename T>
bool col2im(const T* col, const Im2colParams& p, T* output) noexcept {
  static_assert(std::is_arithmetic<T>::value, "col2im requires arithmetic T");
  std::size_t Oh = 0;
  std::size_t Ow = 0;
  if (!im2col_output_dims(p, &Oh, &Ow)) {
    return false;
  }
  // Zero the destination.
  for (std::size_t i = 0; i < p.N * p.C * p.H * p.W; ++i) {
    output[i] = T{0};
  }
  const std::size_t patch_rows = p.C * p.Kh * p.Kw;
  const std::size_t spatial = Oh * Ow;
  for (std::size_t n = 0; n < p.N; ++n) {
    T* out_n = output + n * p.C * p.H * p.W;
    const T* col_n = col + n * patch_rows * spatial;
    for (std::size_t c = 0; c < p.C; ++c) {
      for (std::size_t kh = 0; kh < p.Kh; ++kh) {
        for (std::size_t kw = 0; kw < p.Kw; ++kw) {
          std::size_t row = c * p.Kh * p.Kw + kh * p.Kw + kw;
          const T* col_row = col_n + row * spatial;
          for (std::size_t oh = 0; oh < Oh; ++oh) {
            for (std::size_t ow = 0; ow < Ow; ++ow) {
              std::ptrdiff_t in_h = static_cast<std::ptrdiff_t>(oh * p.Sh +
                                    kh * p.Dh) -
                                    static_cast<std::ptrdiff_t>(p.Ph);
              std::ptrdiff_t in_w = static_cast<std::ptrdiff_t>(ow * p.Sw +
                                    kw * p.Dw) -
                                    static_cast<std::ptrdiff_t>(p.Pw);
              if (in_h >= 0 && in_h < static_cast<std::ptrdiff_t>(p.H) &&
                  in_w >= 0 && in_w < static_cast<std::ptrdiff_t>(p.W)) {
                out_n[c * p.H * p.W +
                      static_cast<std::size_t>(in_h) * p.W +
                      static_cast<std::size_t>(in_w)] +=
                    col_row[oh * Ow + ow];
              }
            }
          }
        }
      }
    }
  }
  return true;
}

}  // namespace zhilicon::kernels::conv
