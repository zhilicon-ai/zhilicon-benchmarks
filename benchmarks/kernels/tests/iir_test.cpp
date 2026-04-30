// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../dsp/iir.hpp"

#include <cmath>
#include <complex>
#include <vector>

using namespace zhilicon::kernels::dsp;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

ZH_TEST(biquad_passthrough_coefficients) {
  // Direct-form biquad with b0 = 1, all others zero is the identity.
  BiquadCoeffs<double> c;
  c.b0 = 1.0;
  Biquad<double> bq(c);
  std::vector<double> input = {1, 2, 3, 4, 5};
  std::vector<double> out(input.size());
  bq.process(input.data(), input.size(), out.data());
  for (std::size_t i = 0; i < input.size(); ++i) {
    ZH_CHECK_NEAR(out[i], input[i], 1e-12);
  }
}

ZH_TEST(biquad_dc_response_lowpass) {
  // Lowpass at 1 kHz cutoff sampled at 48 kHz; DC gain should be 1.0.
  auto c = design_lowpass<double>(48000.0, 1000.0, 0.7071);
  auto h0 = biquad_response(c, 0.0);
  ZH_CHECK_NEAR(std::abs(h0), 1.0, 1e-9);
  // Phase at DC should be ~0 for an even-symmetric numerator.
  ZH_CHECK_NEAR(std::arg(h0), 0.0, 1e-9);
}

ZH_TEST(biquad_nyquist_response_highpass) {
  // Highpass at 1 kHz cutoff sampled at 48 kHz; gain at Nyquist should be 1.
  auto c = design_highpass<double>(48000.0, 1000.0, 0.7071);
  auto h_ny = biquad_response(c, kPi);
  ZH_CHECK_NEAR(std::abs(h_ny), 1.0, 1e-9);
  // DC gain should be ~0.
  auto h0 = biquad_response(c, 0.0);
  ZH_CHECK_NEAR(std::abs(h0), 0.0, 1e-9);
}

ZH_TEST(biquad_bandpass_centre_gain) {
  // Bandpass with Q = 1/sqrt(2) has a peak gain of 1 at the centre.
  auto c = design_bandpass<double>(48000.0, 1000.0, 0.7071);
  double w0 = 2.0 * kPi * 1000.0 / 48000.0;
  auto h = biquad_response(c, w0);
  ZH_CHECK_NEAR(std::abs(h), 1.0, 1e-3);
  auto h0 = biquad_response(c, 0.0);
  ZH_CHECK_NEAR(std::abs(h0), 0.0, 1e-9);
}

ZH_TEST(biquad_lowpass_attenuates_high_freq) {
  auto c = design_lowpass<double>(48000.0, 1000.0, 0.7071);
  // Nyquist should be heavily attenuated.
  auto h_ny = biquad_response(c, kPi);
  ZH_CHECK(std::abs(h_ny) < 0.05);
  // Octave above cutoff should be ~-12 dB (second order).
  double w_octave = 2.0 * kPi * 2000.0 / 48000.0;
  auto h_oct = biquad_response(c, w_octave);
  double mag_db = 20.0 * std::log10(std::abs(h_oct));
  // Expect roughly -12 to -16 dB for a 2nd-order Butterworth lowpass at
  // exactly an octave above cutoff. Allow a 6 dB tolerance because the
  // exact slope depends on Q.
  ZH_CHECK(mag_db < -8.0 && mag_db > -20.0);
}

ZH_TEST(biquad_step_matches_process) {
  auto c = design_lowpass<double>(48000.0, 1000.0, 0.7071);
  Biquad<double> a(c);
  Biquad<double> b(c);
  std::vector<double> input(16);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = std::sin(static_cast<double>(i) * 0.3);
  }
  std::vector<double> out_a(input.size());
  a.process(input.data(), input.size(), out_a.data());
  std::vector<double> out_b(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    out_b[i] = b.step(input[i]);
  }
  for (std::size_t i = 0; i < input.size(); ++i) {
    ZH_CHECK_NEAR(out_a[i], out_b[i], 1e-12);
  }
}

ZH_TEST(biquad_reset_clears_state) {
  auto c = design_lowpass<double>(48000.0, 1000.0, 0.7071);
  Biquad<double> a(c);
  std::vector<double> input = {1, 2, 3, 4, 5};
  std::vector<double> first(input.size());
  std::vector<double> second(input.size());
  a.process(input.data(), input.size(), first.data());
  a.reset();
  a.process(input.data(), input.size(), second.data());
  for (std::size_t i = 0; i < input.size(); ++i) {
    ZH_CHECK_NEAR(first[i], second[i], 1e-12);
  }
}

ZH_TEST(biquad_in_place_processing) {
  auto c = design_lowpass<double>(48000.0, 1000.0, 0.7071);
  Biquad<double> a(c);
  std::vector<double> ref = {1, 2, 3, 4, 5};
  std::vector<double> ref_out(ref.size());
  a.process(ref.data(), ref.size(), ref_out.data());

  Biquad<double> b(c);
  std::vector<double> in_place = ref;
  b.process(in_place.data(), in_place.size(), in_place.data());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    ZH_CHECK_NEAR(in_place[i], ref_out[i], 1e-12);
  }
}

ZH_TEST(biquad_set_coeffs_round_trip) {
  Biquad<float> a;
  BiquadCoeffs<float> c{0.5f, 0.25f, 0.125f, -0.1f, 0.05f};
  a.set_coeffs(c);
  ZH_CHECK_NEAR(a.coeffs().b0, 0.5f, 1e-6);
  ZH_CHECK_NEAR(a.coeffs().b1, 0.25f, 1e-6);
  ZH_CHECK_NEAR(a.coeffs().b2, 0.125f, 1e-6);
  ZH_CHECK_NEAR(a.coeffs().a1, -0.1f, 1e-6);
  ZH_CHECK_NEAR(a.coeffs().a2, 0.05f, 1e-6);
}

ZH_TEST(biquad_lowpass_phase_at_cutoff) {
  // 2nd-order lowpass at cutoff has phase ~ -90 deg.
  auto c = design_lowpass<double>(48000.0, 1000.0, 0.7071);
  double w0 = 2.0 * kPi * 1000.0 / 48000.0;
  auto h = biquad_response(c, w0);
  double phase = std::arg(h);
  ZH_CHECK_NEAR(phase, -kPi / 2.0, 0.05);
}

ZH_TEST(biquad_response_consistency) {
  // Comparing magnitude at 0 and Nyquist for a known transfer function.
  BiquadCoeffs<double> c;
  c.b0 = 1.0;
  c.b1 = -2.0;
  c.b2 = 1.0;
  c.a1 = 0.0;
  c.a2 = 0.0;
  // H(z) = 1 - 2 z^-1 + z^-2 = (1 - z^-1)^2 -> highpass-like.
  auto h0 = biquad_response(c, 0.0);
  ZH_CHECK_NEAR(std::abs(h0), 0.0, 1e-12);
  auto h_ny = biquad_response(c, kPi);
  // 1 + 2 + 1 = 4
  ZH_CHECK_NEAR(std::abs(h_ny), 4.0, 1e-12);
}

ZH_TEST_MAIN("dsp/iir")
