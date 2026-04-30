// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: naive 2-D convolution.
//
// Reference NCHW implementation that loops over every output cell. The
// weight tensor is laid out as (Cout, Cin, Kh, Kw). Stride, padding and
// dilation match the conventions used by PyTorch's torch.nn.functional
// .conv2d (cross-correlation, not flip-and-multiply).
//
// This kernel is intentionally O(N * Cout * Cin * Oh * Ow * Kh * Kw) so
// that it can be used as a numerical oracle for matmul-based or Winograd
// convolution implementations.

#pragma once

#include <cstddef>
#include <type_traits>

namespace zhilicon::kernels::conv {

struct Conv2dParams {
  std::size_t N{1};
  std::size_t Cin{1};
  std::size_t Cout{1};
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

inline bool conv2d_output_dims(const Conv2dParams& p, std::size_t* Oh,
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

// Naive 2-D conv. input layout NCHW. weights layout (Cout, Cin, Kh, Kw).
// bias may be nullptr. output buffer must have N*Cout*Oh*Ow elements.
template <typename T>
bool conv2d_naive(const T* input, const T* weights, const T* bias,
                  const Conv2dParams& p, T* output) noexcept {
  static_assert(std::is_arithmetic<T>::value, "conv2d_naive requires arithmetic T");
  std::size_t Oh = 0;
  std::size_t Ow = 0;
  if (!conv2d_output_dims(p, &Oh, &Ow)) {
    return false;
  }
  for (std::size_t n = 0; n < p.N; ++n) {
    for (std::size_t co = 0; co < p.Cout; ++co) {
      const T b = (bias != nullptr) ? bias[co] : T{0};
      for (std::size_t oh = 0; oh < Oh; ++oh) {
        for (std::size_t ow = 0; ow < Ow; ++ow) {
          T acc = b;
          for (std::size_t ci = 0; ci < p.Cin; ++ci) {
            for (std::size_t kh = 0; kh < p.Kh; ++kh) {
              for (std::size_t kw = 0; kw < p.Kw; ++kw) {
                std::ptrdiff_t in_h = static_cast<std::ptrdiff_t>(oh * p.Sh +
                                      kh * p.Dh) -
                                      static_cast<std::ptrdiff_t>(p.Ph);
                std::ptrdiff_t in_w = static_cast<std::ptrdiff_t>(ow * p.Sw +
                                      kw * p.Dw) -
                                      static_cast<std::ptrdiff_t>(p.Pw);
                if (in_h < 0 || in_h >= static_cast<std::ptrdiff_t>(p.H) ||
                    in_w < 0 || in_w >= static_cast<std::ptrdiff_t>(p.W)) {
                  continue;
                }
                std::size_t in_idx = ((n * p.Cin + ci) * p.H +
                                      static_cast<std::size_t>(in_h)) *
                                     p.W +
                                     static_cast<std::size_t>(in_w);
                std::size_t w_idx = ((co * p.Cin + ci) * p.Kh + kh) * p.Kw + kw;
                acc += input[in_idx] * weights[w_idx];
              }
            }
          }
          std::size_t out_idx = ((n * p.Cout + co) * Oh + oh) * Ow + ow;
          output[out_idx] = acc;
        }
      }
    }
  }
  return true;
}

}  // namespace zhilicon::kernels::conv
