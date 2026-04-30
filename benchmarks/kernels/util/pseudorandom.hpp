// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: deterministic pseudorandom generators.
//
// Benchmarks need reproducible random data so that two runs of the same
// kernel against the same seed produce identical inputs. The standard
// library's std::mt19937 is good enough but exposes too much state for
// inline use; this header provides three lightweight generators with
// deterministic output and well-documented bit-mixing primitives:
//
//   * SplitMix64 - Vigna's seed mixer, useful as a one-shot scrambler.
//   * Xoshiro256** - fast 64-bit generator suitable for benchmark inputs.
//   * uniform_float - converts a 64-bit value into a [0, 1) float.

#pragma once

#include <cstdint>
#include <cstring>

namespace zhilicon::kernels::util {

// SplitMix64: stateless integer hash used to derive Xoshiro state from a
// seed. Reference: http://prng.di.unimi.it/splitmix64.c
inline std::uint64_t splitmix64(std::uint64_t& state) noexcept {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Xoshiro256** PRNG. 256-bit state, period 2^256 - 1.
class Xoshiro256ss {
 public:
  Xoshiro256ss() noexcept { seed(0); }
  explicit Xoshiro256ss(std::uint64_t s) noexcept { seed(s); }

  void seed(std::uint64_t s) noexcept {
    std::uint64_t mix = s;
    s_[0] = splitmix64(mix);
    s_[1] = splitmix64(mix);
    s_[2] = splitmix64(mix);
    s_[3] = splitmix64(mix);
  }

  std::uint64_t next() noexcept {
    std::uint64_t result = rotl(s_[1] * 5ULL, 7) * 9ULL;
    std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  // Fill a buffer with uniformly distributed bytes.
  void fill_bytes(std::uint8_t* dst, std::size_t n) noexcept {
    std::size_t i = 0;
    while (i + 8 <= n) {
      std::uint64_t v = next();
      std::memcpy(dst + i, &v, 8);
      i += 8;
    }
    if (i < n) {
      std::uint64_t v = next();
      std::memcpy(dst + i, &v, n - i);
    }
  }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
  }

  std::uint64_t s_[4]{};
};

// Convert a 64-bit raw value into a uniform float in [0, 1). Uses the
// top 24 bits to fill the fp32 mantissa.
inline float uniform_float_from_u64(std::uint64_t x) noexcept {
  std::uint32_t mant = static_cast<std::uint32_t>(x >> 40) & 0x7FFFFFu;
  // Construct 1.f + mant/2^23, then subtract 1 to get [0, 1).
  std::uint32_t bits = (127u << 23) | mant;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f - 1.0f;
}

// Convert a 64-bit raw value into a uniform double in [0, 1).
inline double uniform_double_from_u64(std::uint64_t x) noexcept {
  std::uint64_t mant = x & ((std::uint64_t{1} << 52) - 1);
  std::uint64_t bits = (std::uint64_t{1023} << 52) | mant;
  double d;
  std::memcpy(&d, &bits, sizeof(d));
  return d - 1.0;
}

// Uniform float in [a, b).
inline float uniform_float(Xoshiro256ss& rng, float a, float b) noexcept {
  return a + (b - a) * uniform_float_from_u64(rng.next());
}

inline double uniform_double(Xoshiro256ss& rng, double a, double b) noexcept {
  return a + (b - a) * uniform_double_from_u64(rng.next());
}

}  // namespace zhilicon::kernels::util
