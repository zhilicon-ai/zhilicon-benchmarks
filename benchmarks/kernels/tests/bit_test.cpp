// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../util/bit.hpp"

#include <cstdint>

using namespace zhilicon::kernels::util;

ZH_TEST(popcount_known_values) {
  ZH_CHECK_EQ(popcount<std::uint32_t>(0), 0);
  ZH_CHECK_EQ(popcount<std::uint32_t>(1), 1);
  ZH_CHECK_EQ(popcount<std::uint32_t>(3), 2);
  ZH_CHECK_EQ(popcount<std::uint32_t>(0xFF), 8);
  ZH_CHECK_EQ(popcount<std::uint32_t>(0xFFFF), 16);
  ZH_CHECK_EQ(popcount<std::uint32_t>(0xFFFFFFFF), 32);
  ZH_CHECK_EQ(popcount<std::uint64_t>(0xFFFFFFFFFFFFFFFFULL), 64);
  ZH_CHECK_EQ(popcount<std::uint64_t>(0xAAAAAAAAAAAAAAAAULL), 32);
}

ZH_TEST(ctz_known_values) {
  ZH_CHECK_EQ(ctz<std::uint32_t>(0), 32);
  ZH_CHECK_EQ(ctz<std::uint32_t>(1), 0);
  ZH_CHECK_EQ(ctz<std::uint32_t>(2), 1);
  ZH_CHECK_EQ(ctz<std::uint32_t>(8), 3);
  ZH_CHECK_EQ(ctz<std::uint32_t>(0x80), 7);
  ZH_CHECK_EQ(ctz<std::uint64_t>(0), 64);
  ZH_CHECK_EQ(ctz<std::uint64_t>(0x100000000ULL), 32);
}

ZH_TEST(clz_known_values) {
  ZH_CHECK_EQ(clz<std::uint32_t>(0), 32);
  ZH_CHECK_EQ(clz<std::uint32_t>(1), 31);
  ZH_CHECK_EQ(clz<std::uint32_t>(2), 30);
  ZH_CHECK_EQ(clz<std::uint32_t>(0x80000000u), 0);
  ZH_CHECK_EQ(clz<std::uint64_t>(0), 64);
  ZH_CHECK_EQ(clz<std::uint64_t>(1), 63);
  ZH_CHECK_EQ(clz<std::uint64_t>(0x8000000000000000ULL), 0);
}

ZH_TEST(bit_reverse_basics) {
  ZH_CHECK_EQ(bit_reverse(0, 4), 0u);
  ZH_CHECK_EQ(bit_reverse(1, 4), 8u);
  ZH_CHECK_EQ(bit_reverse(8, 4), 1u);
  ZH_CHECK_EQ(bit_reverse(0b1011, 4), 0b1101u);
  ZH_CHECK_EQ(bit_reverse(0b00010011, 8), 0b11001000u);
}

ZH_TEST(is_pow2_full_sweep) {
  for (std::uint32_t i = 1; i < 1024; ++i) {
    bool ref = (i & (i - 1)) == 0;
    ZH_CHECK_EQ(is_pow2(i), ref);
  }
  ZH_CHECK(!is_pow2<std::uint32_t>(0));
}

ZH_TEST(next_pow2_known_values) {
  ZH_CHECK_EQ(next_pow2<std::uint32_t>(0), 1u);
  ZH_CHECK_EQ(next_pow2<std::uint32_t>(1), 1u);
  ZH_CHECK_EQ(next_pow2<std::uint32_t>(2), 2u);
  ZH_CHECK_EQ(next_pow2<std::uint32_t>(3), 4u);
  ZH_CHECK_EQ(next_pow2<std::uint32_t>(5), 8u);
  ZH_CHECK_EQ(next_pow2<std::uint32_t>(1023), 1024u);
  ZH_CHECK_EQ(next_pow2<std::uint32_t>(1024), 1024u);
  ZH_CHECK_EQ(next_pow2<std::uint32_t>(1025), 2048u);
}

ZH_TEST(log2_pow2_known_values) {
  ZH_CHECK_EQ(log2_pow2<std::uint32_t>(1), 0);
  ZH_CHECK_EQ(log2_pow2<std::uint32_t>(2), 1);
  ZH_CHECK_EQ(log2_pow2<std::uint32_t>(4), 2);
  ZH_CHECK_EQ(log2_pow2<std::uint32_t>(8), 3);
  ZH_CHECK_EQ(log2_pow2<std::uint32_t>(1024), 10);
  // Non-power-of-two returns -1.
  ZH_CHECK_EQ(log2_pow2<std::uint32_t>(3), -1);
  ZH_CHECK_EQ(log2_pow2<std::uint32_t>(0), -1);
}

ZH_TEST(popcount_clz_consistency) {
  // popcount(2^n) should be 1, clz(2^n) + n + 1 should equal bits.
  for (int n = 0; n < 32; ++n) {
    std::uint32_t v = 1u << n;
    ZH_CHECK_EQ(popcount(v), 1);
    ZH_CHECK_EQ(clz(v) + n + 1, 32);
    ZH_CHECK_EQ(ctz(v), n);
  }
}

ZH_TEST(bit_reverse_involution) {
  // Reversing twice should return the original value.
  for (std::uint32_t v = 0; v < 1024; ++v) {
    ZH_CHECK_EQ(bit_reverse(bit_reverse(v, 10), 10), v);
  }
}

ZH_TEST_MAIN("util/bit")
