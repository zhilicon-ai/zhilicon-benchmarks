// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../util/aligned.hpp"

#include <cstdint>
#include <utility>
#include <vector>

using namespace zhilicon::kernels::util;

ZH_TEST(is_power_of_two_basics) {
  ZH_CHECK(is_power_of_two(1));
  ZH_CHECK(is_power_of_two(2));
  ZH_CHECK(is_power_of_two(64));
  ZH_CHECK(is_power_of_two(1024));
  ZH_CHECK(!is_power_of_two(0));
  ZH_CHECK(!is_power_of_two(3));
  ZH_CHECK(!is_power_of_two(5));
  ZH_CHECK(!is_power_of_two(100));
}

ZH_TEST(round_up_alignment) {
  ZH_CHECK_EQ(round_up(0, 16), std::size_t{0});
  ZH_CHECK_EQ(round_up(1, 16), std::size_t{16});
  ZH_CHECK_EQ(round_up(16, 16), std::size_t{16});
  ZH_CHECK_EQ(round_up(17, 16), std::size_t{32});
  ZH_CHECK_EQ(round_up(63, 64), std::size_t{64});
  ZH_CHECK_EQ(round_up(65, 64), std::size_t{128});
  // Non-power-of-two alignment: returns the input unchanged.
  ZH_CHECK_EQ(round_up(100, 6), std::size_t{100});
}

ZH_TEST(aligned_alloc_returns_aligned_pointer) {
  for (std::size_t align : {std::size_t{16}, std::size_t{32},
                             std::size_t{64}, std::size_t{128},
                             std::size_t{256}, std::size_t{4096}}) {
    void* p = aligned_alloc_bytes(1024, align);
    ZH_CHECK(p != nullptr);
    ZH_CHECK((reinterpret_cast<std::uintptr_t>(p) & (align - 1)) == 0);
    aligned_free(p);
  }
}

ZH_TEST(aligned_alloc_rejects_bad_args) {
  // Non-power-of-two alignment.
  ZH_CHECK(aligned_alloc_bytes(1024, 6) == nullptr);
  // Zero bytes.
  ZH_CHECK(aligned_alloc_bytes(0, 16) == nullptr);
}

ZH_TEST(aligned_buffer_basic_lifecycle) {
  AlignedBuffer<float> buf(1024, 64);
  ZH_CHECK_EQ(buf.size(), std::size_t{1024});
  ZH_CHECK(buf.data() != nullptr);
  ZH_CHECK(is_aligned(buf.data(), 64));
  for (std::size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<float>(i);
  }
  ZH_CHECK_NEAR(buf[100], 100.0f, 1e-6);
  ZH_CHECK_NEAR(buf[1023], 1023.0f, 1e-6);
}

ZH_TEST(aligned_buffer_zero_fill) {
  AlignedBuffer<int> buf(64, 32);
  for (std::size_t i = 0; i < buf.size(); ++i) {
    buf[i] = 7;
  }
  buf.zero();
  for (std::size_t i = 0; i < buf.size(); ++i) {
    ZH_CHECK_EQ(buf[i], 0);
  }
}

ZH_TEST(aligned_buffer_move_constructor) {
  AlignedBuffer<double> a(64, 64);
  a[0] = 3.14;
  void* original = a.data();
  AlignedBuffer<double> b = std::move(a);
  ZH_CHECK_EQ(a.size(), std::size_t{0});
  ZH_CHECK(a.data() == nullptr);
  ZH_CHECK_EQ(b.size(), std::size_t{64});
  ZH_CHECK(b.data() == original);
  ZH_CHECK_NEAR(b[0], 3.14, 1e-12);
}

ZH_TEST(aligned_buffer_move_assignment) {
  AlignedBuffer<int> a(32, 64);
  AlignedBuffer<int> b(16, 64);
  a[0] = 99;
  void* original = a.data();
  b = std::move(a);
  ZH_CHECK_EQ(a.size(), std::size_t{0});
  ZH_CHECK_EQ(b.size(), std::size_t{32});
  ZH_CHECK(b.data() == original);
  ZH_CHECK_EQ(b[0], 99);
}

ZH_TEST(aligned_buffer_release_pointer) {
  AlignedBuffer<int> a(32, 64);
  int* raw = a.release_pointer();
  ZH_CHECK(raw != nullptr);
  ZH_CHECK_EQ(a.size(), std::size_t{0});
  ZH_CHECK(a.data() == nullptr);
  aligned_free(raw);
}

ZH_TEST(is_aligned_helper) {
  alignas(64) int x = 0;
  ZH_CHECK(is_aligned(&x, 64));
  ZH_CHECK(is_aligned(&x, 32));
  // Non-power-of-two: returns false regardless.
  ZH_CHECK(!is_aligned(&x, 24));
}

ZH_TEST(aligned_buffer_default_constructor_empty) {
  AlignedBuffer<int> a;
  ZH_CHECK(a.empty());
  ZH_CHECK(a.data() == nullptr);
  ZH_CHECK_EQ(a.size(), std::size_t{0});
}

ZH_TEST(aligned_buffer_const_data_access) {
  AlignedBuffer<int> a(8, 64);
  for (std::size_t i = 0; i < 8; ++i) {
    a[i] = static_cast<int>(i * 2);
  }
  const AlignedBuffer<int>& cref = a;
  for (std::size_t i = 0; i < 8; ++i) {
    ZH_CHECK_EQ(cref[i], static_cast<int>(i * 2));
  }
  ZH_CHECK(cref.data() != nullptr);
}

ZH_TEST_MAIN("util/aligned")
