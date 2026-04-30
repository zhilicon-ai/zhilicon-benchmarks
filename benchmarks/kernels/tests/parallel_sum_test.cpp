// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../reduce/parallel_sum.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace zhilicon::kernels::reduce;

ZH_TEST(naive_sum_basic) {
  std::vector<float> data = {1, 2, 3, 4, 5};
  ZH_CHECK_NEAR(naive_sum<float>(data.data(), data.size()), 15.0f, 1e-6);
}

ZH_TEST(pairwise_sum_basic) {
  std::vector<float> data = {1, 2, 3, 4, 5};
  ZH_CHECK_NEAR(pairwise_sum<float>(data.data(), data.size()), 15.0f, 1e-6);
}

ZH_TEST(kahan_sum_basic) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  ZH_CHECK_NEAR(kahan_sum<double>(data.data(), data.size()), 15.0, 1e-12);
}

ZH_TEST(empty_input_returns_zero) {
  ZH_CHECK_NEAR(naive_sum<double>(nullptr, 0), 0.0, 0.0);
  ZH_CHECK_NEAR(pairwise_sum<double>(nullptr, 0), 0.0, 0.0);
  ZH_CHECK_NEAR(kahan_sum<double>(nullptr, 0), 0.0, 0.0);
}

ZH_TEST(pairwise_sum_iterative_matches_recursive) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t length : {std::size_t{1}, std::size_t{15},
                             std::size_t{16}, std::size_t{17},
                             std::size_t{1000}, std::size_t{1024}}) {
    std::vector<double> data(length);
    for (auto& v : data) v = dist(rng);
    double a = pairwise_sum<double>(data.data(), data.size());
    double b = pairwise_sum_iterative<double>(data.data(), data.size());
    ZH_CHECK_NEAR(a, b, 1e-12);
  }
}

ZH_TEST(kahan_more_accurate_than_naive_on_pathological) {
  // Construct a sequence with intentional cancellation to break naive sum.
  // 1.0 added many times then -very-large number; pairwise/Kahan should
  // recover better.
  std::vector<double> data;
  data.push_back(1e8);
  for (std::size_t i = 0; i < 10000; ++i) {
    data.push_back(1.0);
  }
  data.push_back(-1e8);
  // True sum = 10000.
  double naive = naive_sum<double>(data.data(), data.size());
  double pairwise = pairwise_sum<double>(data.data(), data.size());
  double kahan = kahan_sum<double>(data.data(), data.size());
  ZH_CHECK_NEAR(kahan, 10000.0, 1e-9);
  // pairwise is between naive and kahan in accuracy.
  ZH_CHECK(std::fabs(pairwise - 10000.0) <= std::fabs(naive - 10000.0) + 1e-9);
}

ZH_TEST(pairwise_sum_large_input) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(-1e-6, 1e-6);
  std::vector<double> data(8192);
  for (auto& v : data) v = dist(rng);
  double a = pairwise_sum<double>(data.data(), data.size());
  double b = naive_sum<double>(data.data(), data.size());
  // The two should agree to roughly 8192 * eps_relative. For this
  // distribution that is comfortably below 1e-9 in absolute terms.
  ZH_CHECK_NEAR(a, b, 1e-9);
}

ZH_TEST(pairwise_mean_basic) {
  std::vector<double> data = {1, 2, 3, 4, 5};
  ZH_CHECK_NEAR(pairwise_mean<double>(data.data(), data.size()), 3.0, 1e-12);
}

ZH_TEST(pairwise_mean_empty_returns_zero) {
  ZH_CHECK_NEAR(pairwise_mean<double>(nullptr, 0), 0.0, 0.0);
}

ZH_TEST(naive_sum_one_element) {
  double v = 42.0;
  ZH_CHECK_NEAR(naive_sum<double>(&v, 1), 42.0, 0.0);
  ZH_CHECK_NEAR(pairwise_sum<double>(&v, 1), 42.0, 0.0);
  ZH_CHECK_NEAR(kahan_sum<double>(&v, 1), 42.0, 0.0);
}

ZH_TEST(pairwise_sum_below_base_case) {
  // Length 8 stays in the iterative base case of pairwise_sum.
  std::vector<float> data = {1, 1, 1, 1, 1, 1, 1, 1};
  ZH_CHECK_NEAR(pairwise_sum<float>(data.data(), data.size()), 8.0f, 0.0f);
}

ZH_TEST(pairwise_sum_above_base_case) {
  // Length 100 exercises recursion.
  std::vector<double> data(100, 1.0);
  ZH_CHECK_NEAR(pairwise_sum<double>(data.data(), data.size()), 100.0, 1e-12);
}

ZH_TEST(pairwise_sum_iterative_odd_length) {
  // Odd length triggers the carry-over branch in the iterative reducer.
  std::vector<double> data(7, 1.0);
  ZH_CHECK_NEAR(pairwise_sum_iterative<double>(data.data(), data.size()), 7.0,
                1e-12);
  std::vector<double> data2(13, 0.5);
  ZH_CHECK_NEAR(pairwise_sum_iterative<double>(data2.data(), data2.size()), 6.5,
                1e-12);
}

ZH_TEST(kahan_sum_alternating_signs) {
  // 1 - 1 + 1 - 1 + ... for 10000 terms = 0.
  std::vector<double> data(10000);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = (i % 2 == 0) ? 1.0 : -1.0;
  }
  double k = kahan_sum<double>(data.data(), data.size());
  ZH_CHECK_NEAR(k, 0.0, 1e-12);
}

ZH_TEST_MAIN("reduce/parallel_sum")
