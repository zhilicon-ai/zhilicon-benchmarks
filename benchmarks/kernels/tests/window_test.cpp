// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../dsp/window.hpp"

#include <cmath>
#include <vector>

using namespace zhilicon::kernels::dsp;

ZH_TEST(rectangular_window_all_ones) {
  std::vector<double> w(64);
  rectangular_window<double>(w.data(), w.size());
  for (auto v : w) {
    ZH_CHECK_NEAR(v, 1.0, 0.0);
  }
}

ZH_TEST(hann_window_endpoints_are_zero) {
  std::vector<double> w(16);
  hann_window<double>(w.data(), w.size());
  ZH_CHECK_NEAR(w[0], 0.0, 1e-12);
  ZH_CHECK_NEAR(w[15], 0.0, 1e-12);
}

ZH_TEST(hann_window_peak_at_centre) {
  std::vector<double> w(17);
  hann_window<double>(w.data(), w.size());
  // Length 17 => index 8 is the centre.
  ZH_CHECK_NEAR(w[8], 1.0, 1e-12);
}

ZH_TEST(hann_window_symmetry) {
  std::vector<double> w(20);
  hann_window<double>(w.data(), w.size());
  for (std::size_t i = 0; i < 10; ++i) {
    ZH_CHECK_NEAR(w[i], w[19 - i], 1e-12);
  }
}

ZH_TEST(hamming_window_endpoints) {
  std::vector<double> w(16);
  hamming_window<double>(w.data(), w.size());
  ZH_CHECK_NEAR(w[0], 0.08, 1e-12);
  ZH_CHECK_NEAR(w[15], 0.08, 1e-12);
}

ZH_TEST(hamming_window_centre) {
  std::vector<double> w(11);
  hamming_window<double>(w.data(), w.size());
  ZH_CHECK_NEAR(w[5], 1.0, 1e-12);
}

ZH_TEST(blackman_window_symmetry) {
  std::vector<double> w(32);
  blackman_window<double>(w.data(), w.size());
  for (std::size_t i = 0; i < 16; ++i) {
    ZH_CHECK_NEAR(w[i], w[31 - i], 1e-12);
  }
}

ZH_TEST(apply_window_multiplies_pointwise) {
  std::vector<double> data(8, 4.0);
  std::vector<double> w(8, 0.5);
  std::vector<double> out(8);
  apply_window<double>(data.data(), w.data(), 8, out.data());
  for (auto v : out) {
    ZH_CHECK_NEAR(v, 2.0, 1e-12);
  }
}

ZH_TEST(window_size_one_returns_one) {
  double w_hann = 0.0;
  double w_hamming = 0.0;
  double w_blackman = 0.0;
  hann_window<double>(&w_hann, 1);
  hamming_window<double>(&w_hamming, 1);
  blackman_window<double>(&w_blackman, 1);
  ZH_CHECK_NEAR(w_hann, 1.0, 0.0);
  ZH_CHECK_NEAR(w_hamming, 1.0, 0.0);
  ZH_CHECK_NEAR(w_blackman, 1.0, 0.0);
}

ZH_TEST(window_size_zero_no_writes) {
  // Sanity: size-0 window does not write anywhere (we pass nullptr).
  hann_window<double>(nullptr, 0);
  hamming_window<double>(nullptr, 0);
  blackman_window<double>(nullptr, 0);
  rectangular_window<double>(nullptr, 0);
  ZH_CHECK(true);
}

ZH_TEST(blackman_window_centre_unity) {
  std::vector<double> w(21);
  blackman_window<double>(w.data(), w.size());
  ZH_CHECK_NEAR(w[10], 1.0, 1e-12);
}

ZH_TEST(hann_known_values_length_5) {
  std::vector<double> w(5);
  hann_window<double>(w.data(), 5);
  // Hann length 5: 0, 0.5, 1, 0.5, 0
  ZH_CHECK_NEAR(w[0], 0.0, 1e-12);
  ZH_CHECK_NEAR(w[1], 0.5, 1e-12);
  ZH_CHECK_NEAR(w[2], 1.0, 1e-12);
  ZH_CHECK_NEAR(w[3], 0.5, 1e-12);
  ZH_CHECK_NEAR(w[4], 0.0, 1e-12);
}

ZH_TEST_MAIN("dsp/window")
