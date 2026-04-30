// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../reduce/argmax.hpp"

#include <vector>

using namespace zhilicon::kernels::reduce;

ZH_TEST(argmax_basic) {
  std::vector<float> data = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
  ZH_CHECK_EQ(argmax<float>(data.data(), data.size()), std::size_t{3});
}

ZH_TEST(argmin_basic) {
  std::vector<float> data = {3.0f, 1.0f, 2.0f, 5.0f, 4.0f};
  ZH_CHECK_EQ(argmin<float>(data.data(), data.size()), std::size_t{1});
}

ZH_TEST(argmax_tie_to_lowest_index) {
  std::vector<int> data = {5, 5, 5, 5};
  ZH_CHECK_EQ(argmax<int>(data.data(), data.size()), std::size_t{0});
}

ZH_TEST(argmin_tie_to_lowest_index) {
  std::vector<int> data = {2, 1, 1, 1};
  ZH_CHECK_EQ(argmin<int>(data.data(), data.size()), std::size_t{1});
}

ZH_TEST(argmax_empty_returns_invalid) {
  ZH_CHECK_EQ(argmax<int>(nullptr, 0), kInvalidIndex);
  ZH_CHECK_EQ(argmin<int>(nullptr, 0), kInvalidIndex);
}

ZH_TEST(argmax_single_element) {
  int v = 42;
  ZH_CHECK_EQ(argmax<int>(&v, 1), std::size_t{0});
  ZH_CHECK_EQ(argmin<int>(&v, 1), std::size_t{0});
}

ZH_TEST(argmax_negative_values) {
  std::vector<double> data = {-5.0, -1.0, -3.0, -2.0};
  ZH_CHECK_EQ(argmax<double>(data.data(), data.size()), std::size_t{1});
  ZH_CHECK_EQ(argmin<double>(data.data(), data.size()), std::size_t{0});
}

ZH_TEST(argmax_first_position) {
  std::vector<int> data = {99, 1, 2, 3};
  ZH_CHECK_EQ(argmax<int>(data.data(), data.size()), std::size_t{0});
}

ZH_TEST(argmax_last_position) {
  std::vector<int> data = {1, 2, 3, 99};
  ZH_CHECK_EQ(argmax<int>(data.data(), data.size()), std::size_t{3});
}

ZH_TEST(topk_returns_top_three) {
  std::vector<int> data = {1, 5, 3, 9, 2, 7, 4};
  std::size_t indices[3];
  std::size_t n = topk_indices<int>(data.data(), data.size(), 3, indices);
  ZH_CHECK_EQ(n, std::size_t{3});
  // Top 3 values are 9 (idx 3), 7 (idx 5), 5 (idx 1).
  ZH_CHECK_EQ(indices[0], std::size_t{3});
  ZH_CHECK_EQ(indices[1], std::size_t{5});
  ZH_CHECK_EQ(indices[2], std::size_t{1});
}

ZH_TEST(topk_handles_k_larger_than_length) {
  std::vector<int> data = {7, 1, 4};
  std::size_t indices[5];
  std::size_t n = topk_indices<int>(data.data(), data.size(), 5, indices);
  ZH_CHECK_EQ(n, std::size_t{3});
  ZH_CHECK_EQ(indices[0], std::size_t{0});  // 7
  ZH_CHECK_EQ(indices[1], std::size_t{2});  // 4
  ZH_CHECK_EQ(indices[2], std::size_t{1});  // 1
}

ZH_TEST(max_with_index_basic) {
  std::vector<float> data = {1.0f, 4.0f, 3.0f, 2.0f};
  auto m = max_with_index<float>(data.data(), data.size());
  ZH_CHECK_NEAR(m.value, 4.0f, 0.0f);
  ZH_CHECK_EQ(m.index, std::size_t{1});
}

ZH_TEST(max_with_index_empty) {
  auto m = max_with_index<float>(nullptr, 0);
  ZH_CHECK_EQ(m.index, kInvalidIndex);
}

ZH_TEST(argmax_with_zero_values) {
  std::vector<int> data = {0, 0, 1, 0};
  ZH_CHECK_EQ(argmax<int>(data.data(), data.size()), std::size_t{2});
}

ZH_TEST(argmax_floating_point_inf) {
  float inf = std::numeric_limits<float>::infinity();
  std::vector<float> data = {1.0f, inf, 3.0f, inf};
  ZH_CHECK_EQ(argmax<float>(data.data(), data.size()), std::size_t{1});
}

ZH_TEST_MAIN("reduce/argmax")
