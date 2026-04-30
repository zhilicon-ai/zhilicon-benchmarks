// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: FIR filter kernel.
//
// Direct-form FIR filter:
//
//     y[n] = sum_{k=0}^{N-1} h[k] * x[n - k]
//
// Three flavours are exposed:
//   * fir_apply   - non-causal symmetric input (input length must be >= taps)
//   * fir_apply_zero_pad - input is zero-padded on the left, so the first
//                          tap_count - 1 outputs reflect the warmup tail
//   * fir_apply_streaming - stateful filter that retains the tail buffer
//                           across invocations for streaming benchmarks
//
// The kernel is templated on Sample (float or double). It is portable C++17
// with no compiler intrinsics; vectorisation is left to the compiler.

#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace zhilicon::kernels::dsp {

// Apply FIR filter, treating values before x[0] as zero. Output length
// matches input length. taps[0] is the most recent coefficient.
template <typename Sample>
void fir_apply_zero_pad(const Sample* taps, std::size_t tap_count,
                        const Sample* input, std::size_t length,
                        Sample* output) noexcept {
  static_assert(std::is_floating_point<Sample>::value,
                "fir_apply_zero_pad requires a floating point sample type");
  for (std::size_t n = 0; n < length; ++n) {
    Sample acc = Sample{0};
    const std::size_t kmax = (n + 1) < tap_count ? (n + 1) : tap_count;
    for (std::size_t k = 0; k < kmax; ++k) {
      acc += taps[k] * input[n - k];
    }
    output[n] = acc;
  }
}

// Apply FIR filter assuming input is fully populated. Output length is
// length - tap_count + 1; caller must size the output buffer accordingly.
// Returns the number of valid output samples produced. Returns 0 if the
// input is shorter than tap_count or the inputs are inconsistent.
template <typename Sample>
std::size_t fir_apply(const Sample* taps, std::size_t tap_count,
                      const Sample* input, std::size_t length,
                      Sample* output) noexcept {
  static_assert(std::is_floating_point<Sample>::value,
                "fir_apply requires a floating point sample type");
  if (tap_count == 0 || length < tap_count) {
    return 0;
  }
  const std::size_t out_len = length - tap_count + 1;
  for (std::size_t n = 0; n < out_len; ++n) {
    Sample acc = Sample{0};
    // Convolve, with the tap order reversed to match the math definition.
    const Sample* xp = input + n + tap_count - 1;
    for (std::size_t k = 0; k < tap_count; ++k) {
      acc += taps[k] * xp[-static_cast<std::ptrdiff_t>(k)];
    }
    output[n] = acc;
  }
  return out_len;
}

// Stateful streaming FIR filter. Maintains a delay line so that calls to
// process() can be chained block-by-block.
template <typename Sample>
class StreamingFir {
  static_assert(std::is_floating_point<Sample>::value,
                "StreamingFir requires a floating point sample type");

 public:
  StreamingFir() = default;

  StreamingFir(const Sample* taps, std::size_t tap_count)
      : taps_(taps, taps + tap_count), state_(tap_count, Sample{0}) {}

  void set_taps(const Sample* taps, std::size_t tap_count) {
    taps_.assign(taps, taps + tap_count);
    state_.assign(tap_count, Sample{0});
  }

  // Reset the delay line to all zeros, leaving the tap coefficients
  // untouched.
  void reset() noexcept {
    std::fill(state_.begin(), state_.end(), Sample{0});
  }

  std::size_t tap_count() const noexcept { return taps_.size(); }

  // Process a block of input samples. Output buffer length must match input
  // length. Returns the number of samples processed.
  std::size_t process(const Sample* input, std::size_t length,
                      Sample* output) noexcept {
    if (taps_.empty()) {
      return 0;
    }
    const std::size_t taps = taps_.size();
    for (std::size_t i = 0; i < length; ++i) {
      // Shift state right by one and insert new sample at index 0.
      for (std::size_t k = taps - 1; k > 0; --k) {
        state_[k] = state_[k - 1];
      }
      state_[0] = input[i];
      Sample acc = Sample{0};
      for (std::size_t k = 0; k < taps; ++k) {
        acc += taps_[k] * state_[k];
      }
      output[i] = acc;
    }
    return length;
  }

 private:
  std::vector<Sample> taps_;
  std::vector<Sample> state_;
};

// Convenience: convolve two arrays of equal length using zero-padded FIR.
// This is useful for small reference implementations in tests.
template <typename Sample>
void direct_convolve(const Sample* a, std::size_t a_len, const Sample* b,
                     std::size_t b_len, Sample* output) noexcept {
  const std::size_t out_len = a_len + b_len - 1;
  for (std::size_t n = 0; n < out_len; ++n) {
    Sample acc = Sample{0};
    const std::size_t kmin = (n >= b_len - 1) ? (n - (b_len - 1)) : 0;
    const std::size_t kmax = (n < a_len - 1) ? n : (a_len - 1);
    for (std::size_t k = kmin; k <= kmax; ++k) {
      acc += a[k] * b[n - k];
    }
    output[n] = acc;
  }
}

}  // namespace zhilicon::kernels::dsp
