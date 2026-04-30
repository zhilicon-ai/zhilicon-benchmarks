// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../util/pseudorandom.hpp"

#include <array>
#include <cmath>

using namespace zhilicon::kernels::util;

ZH_TEST(splitmix64_deterministic) {
  std::uint64_t a = 42;
  std::uint64_t b = 42;
  for (int i = 0; i < 16; ++i) {
    ZH_CHECK_EQ(splitmix64(a), splitmix64(b));
  }
}

ZH_TEST(splitmix64_non_zero_for_zero_state) {
  std::uint64_t s = 0;
  std::uint64_t v = splitmix64(s);
  ZH_CHECK(v != 0);
}

ZH_TEST(xoshiro_deterministic_with_seed) {
  Xoshiro256ss a(123);
  Xoshiro256ss b(123);
  for (int i = 0; i < 32; ++i) {
    ZH_CHECK_EQ(a.next(), b.next());
  }
}

ZH_TEST(xoshiro_different_seeds_diverge) {
  Xoshiro256ss a(0);
  Xoshiro256ss b(1);
  bool any_diff = false;
  for (int i = 0; i < 4; ++i) {
    if (a.next() != b.next()) {
      any_diff = true;
      break;
    }
  }
  ZH_CHECK(any_diff);
}

ZH_TEST(xoshiro_fill_bytes_full_block) {
  Xoshiro256ss rng(99);
  std::array<std::uint8_t, 32> buf{};
  rng.fill_bytes(buf.data(), buf.size());
  // Check that not every byte is zero (extremely unlikely with a real PRNG).
  bool any = false;
  for (auto b : buf) {
    if (b != 0) {
      any = true;
      break;
    }
  }
  ZH_CHECK(any);
}

ZH_TEST(xoshiro_fill_bytes_partial_tail) {
  Xoshiro256ss rng(42);
  std::array<std::uint8_t, 13> buf{};
  rng.fill_bytes(buf.data(), buf.size());
  bool any = false;
  for (auto b : buf) {
    if (b != 0) {
      any = true;
      break;
    }
  }
  ZH_CHECK(any);
}

ZH_TEST(uniform_float_in_range) {
  Xoshiro256ss rng(7);
  for (int i = 0; i < 1024; ++i) {
    float v = uniform_float(rng, 0.0f, 1.0f);
    ZH_CHECK(v >= 0.0f);
    ZH_CHECK(v < 1.0f);
  }
}

ZH_TEST(uniform_float_custom_range) {
  Xoshiro256ss rng(11);
  for (int i = 0; i < 1024; ++i) {
    float v = uniform_float(rng, -5.0f, 5.0f);
    ZH_CHECK(v >= -5.0f);
    ZH_CHECK(v < 5.0f);
  }
}

ZH_TEST(uniform_double_in_range) {
  Xoshiro256ss rng(13);
  for (int i = 0; i < 1024; ++i) {
    double v = uniform_double(rng, 0.0, 1.0);
    ZH_CHECK(v >= 0.0);
    ZH_CHECK(v < 1.0);
  }
}

ZH_TEST(uniform_float_mean_close_to_centre) {
  Xoshiro256ss rng(0);
  double sum = 0.0;
  const int n = 100000;
  for (int i = 0; i < n; ++i) {
    sum += static_cast<double>(uniform_float(rng, 0.0f, 1.0f));
  }
  double mean = sum / n;
  // Expect mean around 0.5; allow generous tolerance.
  ZH_CHECK(std::fabs(mean - 0.5) < 0.02);
}

ZH_TEST(default_constructor_repeats_seed_zero) {
  Xoshiro256ss a;
  Xoshiro256ss b(0);
  for (int i = 0; i < 8; ++i) {
    ZH_CHECK_EQ(a.next(), b.next());
  }
}

ZH_TEST(uniform_double_from_u64_zero_input) {
  // A zero-bit input must produce 0.0 (smallest representable value).
  double v = uniform_double_from_u64(0);
  ZH_CHECK_NEAR(v, 0.0, 0.0);
}

ZH_TEST(uniform_float_from_u64_zero_input) {
  float v = uniform_float_from_u64(0);
  ZH_CHECK_NEAR(v, 0.0f, 0.0f);
}

ZH_TEST_MAIN("util/pseudorandom")
