// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: argmax / argmin reductions with stable tie-breaking.
//
// Tie-breaking convention: the lowest index wins. This matches numpy's
// argmax behaviour and is the convention we have committed to in the
// methodology document for accuracy comparisons.
//
// For empty inputs the routines return the sentinel kInvalidIndex and
// callers should treat it as an error.

#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

namespace zhilicon::kernels::reduce {

constexpr std::size_t kInvalidIndex = static_cast<std::size_t>(-1);

// Returns the index of the largest element, breaking ties to the lowest
// index.
template <typename T>
std::size_t argmax(const T* data, std::size_t length) noexcept {
  if (length == 0) {
    return kInvalidIndex;
  }
  std::size_t best_idx = 0;
  T best_val = data[0];
  for (std::size_t i = 1; i < length; ++i) {
    if (data[i] > best_val) {
      best_val = data[i];
      best_idx = i;
    }
  }
  return best_idx;
}

// Returns the index of the smallest element, breaking ties to the lowest
// index.
template <typename T>
std::size_t argmin(const T* data, std::size_t length) noexcept {
  if (length == 0) {
    return kInvalidIndex;
  }
  std::size_t best_idx = 0;
  T best_val = data[0];
  for (std::size_t i = 1; i < length; ++i) {
    if (data[i] < best_val) {
      best_val = data[i];
      best_idx = i;
    }
  }
  return best_idx;
}

// Top-k (largest values) descending. Returns indices into data, with the
// strongest match at indices[0]. Tie-breaking: lower index wins. The
// caller must size `indices` to at least k. Returns the number of valid
// entries written (min(k, length)). The implementation uses a partial
// selection sort for clarity; for the small k used in benchmarks (1, 5,
// 10) this is competitive with priority-queue alternatives and produces
// deterministic output.
template <typename T>
std::size_t topk_indices(const T* data, std::size_t length, std::size_t k,
                         std::size_t* indices) noexcept {
  std::size_t out = (k < length) ? k : length;
  for (std::size_t i = 0; i < out; ++i) {
    std::size_t best = kInvalidIndex;
    T best_val{};
    for (std::size_t j = 0; j < length; ++j) {
      bool already_chosen = false;
      for (std::size_t m = 0; m < i; ++m) {
        if (indices[m] == j) {
          already_chosen = true;
          break;
        }
      }
      if (already_chosen) continue;
      if (best == kInvalidIndex || data[j] > best_val ||
          (data[j] == best_val && j < best)) {
        if (best == kInvalidIndex) {
          best = j;
          best_val = data[j];
        } else if (data[j] > best_val) {
          best = j;
          best_val = data[j];
        }
      }
    }
    indices[i] = best;
  }
  return out;
}

// Variant returning both the maximum value and its index.
template <typename T>
struct MaxResult {
  T value{};
  std::size_t index{kInvalidIndex};
};

template <typename T>
MaxResult<T> max_with_index(const T* data, std::size_t length) noexcept {
  MaxResult<T> r;
  if (length == 0) {
    return r;
  }
  r.value = data[0];
  r.index = 0;
  for (std::size_t i = 1; i < length; ++i) {
    if (data[i] > r.value) {
      r.value = data[i];
      r.index = i;
    }
  }
  return r;
}

}  // namespace zhilicon::kernels::reduce
