// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../reduce/variance.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace zhilicon::kernels::reduce;

ZH_TEST(online_mean_var_constant_signal) {
  std::vector<double> data(100, 7.0);
  auto m = online_mean_var<double>(data.data(), data.size());
  ZH_CHECK_NEAR(m.mean, 7.0, 1e-12);
  ZH_CHECK_NEAR(m.m2, 0.0, 1e-12);
  ZH_CHECK_EQ(m.count, std::size_t{100});
}

ZH_TEST(sample_variance_known_set) {
  // [2, 4, 4, 4, 5, 5, 7, 9]: mean = 5, sum of sq deviations = 32.
  // Sample var = 32 / (8-1) = 32/7
  std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
  ZH_CHECK_NEAR(sample_variance<double>(data.data(), data.size()),
                32.0 / 7.0, 1e-12);
}

ZH_TEST(population_variance_known_set) {
  // Same data: population var = 32/8 = 4
  std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
  ZH_CHECK_NEAR(population_variance<double>(data.data(), data.size()), 4.0,
                1e-12);
}

ZH_TEST(sample_variance_returns_zero_for_lt_2) {
  std::vector<double> single = {3.14};
  ZH_CHECK_NEAR(sample_variance<double>(single.data(), single.size()), 0.0,
                0.0);
  ZH_CHECK_NEAR(sample_variance<double>(nullptr, 0), 0.0, 0.0);
}

ZH_TEST(population_variance_returns_zero_for_empty) {
  ZH_CHECK_NEAR(population_variance<double>(nullptr, 0), 0.0, 0.0);
}

ZH_TEST(stddev_matches_sqrt_var) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  double var = sample_variance<double>(data.data(), data.size());
  double sd = sample_stddev<double>(data.data(), data.size());
  ZH_CHECK_NEAR(sd, std::sqrt(var), 1e-12);
}

ZH_TEST(merge_two_halves_matches_single_pass) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> all(64);
  for (auto& v : all) v = dist(rng);
  auto full = online_mean_var<double>(all.data(), all.size());
  auto first = online_mean_var<double>(all.data(), 30);
  auto second = online_mean_var<double>(all.data() + 30, 34);
  auto merged = merge(first, second);
  ZH_CHECK_EQ(merged.count, full.count);
  ZH_CHECK_NEAR(merged.mean, full.mean, 1e-9);
  ZH_CHECK_NEAR(merged.m2, full.m2, 1e-6);
}

ZH_TEST(merge_with_empty_either_side_returns_other) {
  MeanVar<double> empty;
  MeanVar<double> a;
  a.mean = 5.0;
  a.m2 = 10.0;
  a.count = 4;
  auto m1 = merge(empty, a);
  auto m2 = merge(a, empty);
  ZH_CHECK_EQ(m1.count, std::size_t{4});
  ZH_CHECK_NEAR(m1.mean, 5.0, 1e-12);
  ZH_CHECK_NEAR(m1.m2, 10.0, 1e-12);
  ZH_CHECK_EQ(m2.count, std::size_t{4});
  ZH_CHECK_NEAR(m2.mean, 5.0, 1e-12);
}

ZH_TEST(welford_avoids_catastrophic_cancellation) {
  // Samples drawn near a large mean. Welford should still report the
  // correct variance to within float-size epsilon.
  std::vector<double> data;
  for (int i = 0; i < 1000; ++i) {
    data.push_back(1e9 + static_cast<double>(i % 7));
  }
  double var = sample_variance<double>(data.data(), data.size());
  // Variance of a uniform set {0..6} repeated is 4.0; with the N-1
  // denominator and 1000 samples the value should be very close.
  ZH_CHECK_NEAR(var, 4.0, 1e-2);
}

ZH_TEST(population_stddev_known_value) {
  std::vector<double> data = {0, 0, 0, 0, 4};
  // Mean = 0.8, deviations: -0.8, -0.8, -0.8, -0.8, 3.2
  // var = (0.64*4 + 10.24)/5 = (2.56 + 10.24)/5 = 2.56
  // std = 1.6
  double sd = population_stddev<double>(data.data(), data.size());
  ZH_CHECK_NEAR(sd, 1.6, 1e-12);
}

ZH_TEST(online_update_one_sample) {
  MeanVar<double> acc;
  mean_var_update<double>(acc, 42.0);
  ZH_CHECK_EQ(acc.count, std::size_t{1});
  ZH_CHECK_NEAR(acc.mean, 42.0, 1e-12);
  ZH_CHECK_NEAR(acc.m2, 0.0, 1e-12);
}

ZH_TEST(merge_three_partitions) {
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {4, 5};
  std::vector<double> c = {6, 7, 8, 9};
  std::vector<double> all = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto ma = online_mean_var<double>(a.data(), a.size());
  auto mb = online_mean_var<double>(b.data(), b.size());
  auto mc = online_mean_var<double>(c.data(), c.size());
  auto m_all = online_mean_var<double>(all.data(), all.size());
  auto merged = merge(merge(ma, mb), mc);
  ZH_CHECK_NEAR(merged.mean, m_all.mean, 1e-12);
  ZH_CHECK_NEAR(merged.m2, m_all.m2, 1e-9);
}

ZH_TEST_MAIN("reduce/variance")
