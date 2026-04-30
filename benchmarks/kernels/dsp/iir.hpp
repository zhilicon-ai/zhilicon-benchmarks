// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: 2nd-order biquad IIR filter (Direct Form II Transposed).
//
// The biquad realises the transfer function
//
//     H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
//
// We adopt the convention that the leading 'a0' is normalised to 1 by the
// caller. Coefficients are passed via a struct so that filter banks can be
// laid out cache-friendly in memory.
//
// Direct Form II Transposed is numerically the most stable single-section
// biquad realisation; see Smith, "Introduction to Digital Filters", chapter
// on biquad implementations.

#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>

namespace zhilicon::kernels::dsp {

template <typename Sample>
struct BiquadCoeffs {
  static_assert(std::is_floating_point<Sample>::value,
                "BiquadCoeffs requires a floating point sample type");
  Sample b0{1};
  Sample b1{0};
  Sample b2{0};
  Sample a1{0};
  Sample a2{0};
};

template <typename Sample>
class Biquad {
  static_assert(std::is_floating_point<Sample>::value,
                "Biquad requires a floating point sample type");

 public:
  Biquad() = default;
  explicit Biquad(const BiquadCoeffs<Sample>& c) noexcept : c_(c) {}

  void set_coeffs(const BiquadCoeffs<Sample>& c) noexcept { c_ = c; }
  const BiquadCoeffs<Sample>& coeffs() const noexcept { return c_; }

  void reset() noexcept {
    z1_ = Sample{0};
    z2_ = Sample{0};
  }

  // Apply to a buffer in-place or out-of-place. Output may alias input.
  void process(const Sample* input, std::size_t length, Sample* output) noexcept {
    Sample z1 = z1_;
    Sample z2 = z2_;
    for (std::size_t i = 0; i < length; ++i) {
      Sample x = input[i];
      Sample y = c_.b0 * x + z1;
      z1 = c_.b1 * x - c_.a1 * y + z2;
      z2 = c_.b2 * x - c_.a2 * y;
      output[i] = y;
    }
    z1_ = z1;
    z2_ = z2;
  }

  // Single-sample step. Useful in tests where we want to feed an impulse
  // and collect the response sample by sample.
  Sample step(Sample x) noexcept {
    Sample y = c_.b0 * x + z1_;
    z1_ = c_.b1 * x - c_.a1 * y + z2_;
    z2_ = c_.b2 * x - c_.a2 * y;
    return y;
  }

 private:
  BiquadCoeffs<Sample> c_{};
  Sample z1_{0};
  Sample z2_{0};
};

// Evaluate H(z) at z = exp(j * omega) for the given biquad. omega is in
// radians/sample. Returns complex H value in fp64 precision regardless of
// Sample to keep numerical analysis tests stable.
template <typename Sample>
std::complex<double> biquad_response(const BiquadCoeffs<Sample>& c,
                                     double omega) noexcept {
  const std::complex<double> z_inv(std::cos(-omega), std::sin(-omega));
  std::complex<double> num(static_cast<double>(c.b0));
  num += static_cast<double>(c.b1) * z_inv;
  num += static_cast<double>(c.b2) * z_inv * z_inv;
  std::complex<double> den(1.0);
  den += static_cast<double>(c.a1) * z_inv;
  den += static_cast<double>(c.a2) * z_inv * z_inv;
  return num / den;
}

// Design helpers (RBJ Audio EQ Cookbook). Each returns coefficients
// normalised so that a0 == 1.
template <typename Sample>
BiquadCoeffs<Sample> design_lowpass(double sample_rate, double cutoff_hz,
                                    double q) noexcept {
  const double w0 = 2.0 * 3.14159265358979323846 * cutoff_hz / sample_rate;
  const double cosw0 = std::cos(w0);
  const double sinw0 = std::sin(w0);
  const double alpha = sinw0 / (2.0 * q);

  const double b0 = (1.0 - cosw0) / 2.0;
  const double b1 = 1.0 - cosw0;
  const double b2 = (1.0 - cosw0) / 2.0;
  const double a0 = 1.0 + alpha;
  const double a1 = -2.0 * cosw0;
  const double a2 = 1.0 - alpha;

  BiquadCoeffs<Sample> out;
  out.b0 = static_cast<Sample>(b0 / a0);
  out.b1 = static_cast<Sample>(b1 / a0);
  out.b2 = static_cast<Sample>(b2 / a0);
  out.a1 = static_cast<Sample>(a1 / a0);
  out.a2 = static_cast<Sample>(a2 / a0);
  return out;
}

template <typename Sample>
BiquadCoeffs<Sample> design_highpass(double sample_rate, double cutoff_hz,
                                     double q) noexcept {
  const double w0 = 2.0 * 3.14159265358979323846 * cutoff_hz / sample_rate;
  const double cosw0 = std::cos(w0);
  const double sinw0 = std::sin(w0);
  const double alpha = sinw0 / (2.0 * q);

  const double b0 = (1.0 + cosw0) / 2.0;
  const double b1 = -(1.0 + cosw0);
  const double b2 = (1.0 + cosw0) / 2.0;
  const double a0 = 1.0 + alpha;
  const double a1 = -2.0 * cosw0;
  const double a2 = 1.0 - alpha;

  BiquadCoeffs<Sample> out;
  out.b0 = static_cast<Sample>(b0 / a0);
  out.b1 = static_cast<Sample>(b1 / a0);
  out.b2 = static_cast<Sample>(b2 / a0);
  out.a1 = static_cast<Sample>(a1 / a0);
  out.a2 = static_cast<Sample>(a2 / a0);
  return out;
}

template <typename Sample>
BiquadCoeffs<Sample> design_bandpass(double sample_rate, double centre_hz,
                                     double q) noexcept {
  const double w0 = 2.0 * 3.14159265358979323846 * centre_hz / sample_rate;
  const double cosw0 = std::cos(w0);
  const double sinw0 = std::sin(w0);
  const double alpha = sinw0 / (2.0 * q);

  const double b0 = alpha;
  const double b1 = 0.0;
  const double b2 = -alpha;
  const double a0 = 1.0 + alpha;
  const double a1 = -2.0 * cosw0;
  const double a2 = 1.0 - alpha;

  BiquadCoeffs<Sample> out;
  out.b0 = static_cast<Sample>(b0 / a0);
  out.b1 = static_cast<Sample>(b1 / a0);
  out.b2 = static_cast<Sample>(b2 / a0);
  out.a1 = static_cast<Sample>(a1 / a0);
  out.a2 = static_cast<Sample>(a2 / a0);
  return out;
}

}  // namespace zhilicon::kernels::dsp
