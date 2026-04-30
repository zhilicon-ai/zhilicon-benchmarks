// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../dsp/fft_radix2.hpp"

#include <cmath>
#include <complex>
#include <vector>

using namespace zhilicon::kernels::dsp;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

ZH_TEST(fft_constant_signal) {
  // FFT of N constant samples => N at bin 0, zero elsewhere.
  const std::size_t N = 16;
  std::vector<std::complex<double>> data(N, std::complex<double>(1.0, 0.0));
  ZH_CHECK(fft_radix2_inplace(data.data(), N));
  ZH_CHECK_NEAR(data[0].real(), 16.0, 1e-9);
  ZH_CHECK_NEAR(data[0].imag(), 0.0, 1e-9);
  for (std::size_t i = 1; i < N; ++i) {
    ZH_CHECK_NEAR(std::abs(data[i]), 0.0, 1e-9);
  }
}

ZH_TEST(fft_rejects_non_power_of_two) {
  std::vector<std::complex<double>> data(15, {1.0, 0.0});
  ZH_CHECK(!fft_radix2_inplace(data.data(), 15));
}

ZH_TEST(fft_size_one_is_identity) {
  std::vector<std::complex<float>> data = {{3.0f, 4.0f}};
  ZH_CHECK(fft_radix2_inplace(data.data(), 1));
  ZH_CHECK_NEAR(data[0].real(), 3.0f, 1e-6);
  ZH_CHECK_NEAR(data[0].imag(), 4.0f, 1e-6);
}

ZH_TEST(fft_matches_naive_dft_size_8) {
  const std::size_t N = 8;
  std::vector<std::complex<double>> input(N);
  for (std::size_t i = 0; i < N; ++i) {
    input[i] = {std::sin(0.4 * static_cast<double>(i)),
                std::cos(0.7 * static_cast<double>(i))};
  }
  std::vector<std::complex<double>> ref(N);
  naive_dft<double>(input.data(), N, ref.data());

  std::vector<std::complex<double>> got = input;
  ZH_CHECK(fft_radix2_inplace(got.data(), N));
  for (std::size_t i = 0; i < N; ++i) {
    ZH_CHECK_NEAR(got[i].real(), ref[i].real(), 1e-9);
    ZH_CHECK_NEAR(got[i].imag(), ref[i].imag(), 1e-9);
  }
}

ZH_TEST(fft_matches_naive_dft_size_64) {
  const std::size_t N = 64;
  std::vector<std::complex<double>> input(N);
  for (std::size_t i = 0; i < N; ++i) {
    input[i] = {std::cos(2.0 * kPi * 3.0 * static_cast<double>(i) / N),
                0.0};
  }
  std::vector<std::complex<double>> ref(N);
  naive_dft<double>(input.data(), N, ref.data());

  std::vector<std::complex<double>> got = input;
  ZH_CHECK(fft_radix2_inplace(got.data(), N));
  for (std::size_t i = 0; i < N; ++i) {
    ZH_CHECK_NEAR(got[i].real(), ref[i].real(), 1e-9);
    ZH_CHECK_NEAR(got[i].imag(), ref[i].imag(), 1e-9);
  }
}

ZH_TEST(fft_inverse_round_trip) {
  const std::size_t N = 32;
  std::vector<std::complex<double>> input(N);
  for (std::size_t i = 0; i < N; ++i) {
    input[i] = {static_cast<double>(i) - 16.0,
                static_cast<double>((i * 3) % 11)};
  }
  std::vector<std::complex<double>> data = input;
  ZH_CHECK(fft_radix2_inplace(data.data(), N));
  ZH_CHECK(ifft_radix2_inplace(data.data(), N));
  for (std::size_t i = 0; i < N; ++i) {
    ZH_CHECK_NEAR(data[i].real(), input[i].real(), 1e-9);
    ZH_CHECK_NEAR(data[i].imag(), input[i].imag(), 1e-9);
  }
}

ZH_TEST(fft_single_bin_impulse) {
  // FFT of a kronecker delta x[n] = delta[n] yields ones in every bin.
  const std::size_t N = 8;
  std::vector<std::complex<double>> data(N, {0.0, 0.0});
  data[0] = {1.0, 0.0};
  ZH_CHECK(fft_radix2_inplace(data.data(), N));
  for (std::size_t i = 0; i < N; ++i) {
    ZH_CHECK_NEAR(data[i].real(), 1.0, 1e-9);
    ZH_CHECK_NEAR(data[i].imag(), 0.0, 1e-9);
  }
}

ZH_TEST(fft_complex_exponential_picks_one_bin) {
  // Pure complex exponential at bin k should have all energy in bin k.
  const std::size_t N = 16;
  const std::size_t k = 3;
  std::vector<std::complex<double>> data(N);
  for (std::size_t i = 0; i < N; ++i) {
    double theta = 2.0 * kPi * static_cast<double>(k) *
                   static_cast<double>(i) / static_cast<double>(N);
    data[i] = {std::cos(theta), std::sin(theta)};
  }
  ZH_CHECK(fft_radix2_inplace(data.data(), N));
  for (std::size_t i = 0; i < N; ++i) {
    if (i == k) {
      ZH_CHECK_NEAR(data[i].real(), static_cast<double>(N), 1e-9);
      ZH_CHECK_NEAR(data[i].imag(), 0.0, 1e-9);
    } else {
      ZH_CHECK(std::abs(data[i]) < 1e-9);
    }
  }
}

ZH_TEST(fft_plan_matches_inplace) {
  const std::size_t N = 64;
  std::vector<std::complex<float>> base(N);
  for (std::size_t i = 0; i < N; ++i) {
    base[i] = {static_cast<float>(i % 7),
               static_cast<float>((i * 13) % 11)};
  }
  std::vector<std::complex<float>> a = base;
  std::vector<std::complex<float>> b = base;
  ZH_CHECK(fft_radix2_inplace(a.data(), N));
  FftPlan<float> plan(N);
  ZH_CHECK_EQ(plan.size(), N);
  ZH_CHECK(plan.execute(b.data()));
  for (std::size_t i = 0; i < N; ++i) {
    ZH_CHECK_NEAR(a[i].real(), b[i].real(), 1e-4);
    ZH_CHECK_NEAR(a[i].imag(), b[i].imag(), 1e-4);
  }
}

ZH_TEST(fft_plan_rejects_non_power_of_two) {
  FftPlan<double> plan;
  ZH_CHECK(!plan.resize(15));
  ZH_CHECK(plan.resize(16));
  ZH_CHECK_EQ(plan.size(), std::size_t{16});
}

ZH_TEST(fft_size_2_known_values) {
  // [a, b] -> [a + b, a - b]
  std::vector<std::complex<double>> data = {{3.0, 0.0}, {1.0, 0.0}};
  ZH_CHECK(fft_radix2_inplace(data.data(), 2));
  ZH_CHECK_NEAR(data[0].real(), 4.0, 1e-12);
  ZH_CHECK_NEAR(data[1].real(), 2.0, 1e-12);
}

ZH_TEST(fft_size_4_known_values) {
  // FFT of [1, 0, 0, 0] -> all ones.
  std::vector<std::complex<double>> data = {{1, 0}, {0, 0}, {0, 0}, {0, 0}};
  ZH_CHECK(fft_radix2_inplace(data.data(), 4));
  for (auto& v : data) {
    ZH_CHECK_NEAR(v.real(), 1.0, 1e-12);
    ZH_CHECK_NEAR(v.imag(), 0.0, 1e-12);
  }
}

ZH_TEST(ifft_rejects_non_power_of_two) {
  std::vector<std::complex<double>> data(15);
  ZH_CHECK(!ifft_radix2_inplace(data.data(), 15));
}

ZH_TEST(fft_float_precision_basic) {
  const std::size_t N = 8;
  std::vector<std::complex<float>> a(N);
  for (std::size_t i = 0; i < N; ++i) {
    a[i] = {1.0f, 0.0f};
  }
  ZH_CHECK(fft_radix2_inplace(a.data(), N));
  ZH_CHECK_NEAR(a[0].real(), 8.0f, 1e-5);
  for (std::size_t i = 1; i < N; ++i) {
    ZH_CHECK(static_cast<double>(std::abs(a[i])) < 1e-5);
  }
}

ZH_TEST_MAIN("dsp/fft")
