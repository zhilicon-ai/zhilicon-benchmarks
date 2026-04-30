// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: aligned allocation utilities.
//
// Provides a portable RAII wrapper around aligned heap allocation suitable
// for SIMD-friendly buffers in the benchmark kernels. The allocator targets
// 64-byte alignment by default (cache line on most modern CPUs and the
// natural alignment for AVX-512 / SVE-256 working sets) but accepts any
// power-of-two alignment.
//
// Implementation notes:
//   * The header is self-contained; only <cstddef>, <cstdlib>, <cstring>,
//     <memory>, <new>, and <type_traits> are pulled in.
//   * Allocation never throws; failures surface as a null pointer.
//   * The buffer object is movable but not copyable, mirroring std::vector
//     semantics for owning resources.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace zhilicon::kernels::util {

// Returns true if x is a power of two (and non-zero).
constexpr bool is_power_of_two(std::size_t x) noexcept {
  return x != 0 && (x & (x - 1)) == 0;
}

// Round n up to the next multiple of alignment. alignment must be a power of
// two (callers should assert this; in release builds undefined alignment
// returns n unchanged).
constexpr std::size_t round_up(std::size_t n, std::size_t alignment) noexcept {
  if (!is_power_of_two(alignment)) {
    return n;
  }
  return (n + alignment - 1) & ~(alignment - 1);
}

// Aligned allocation. Returns nullptr on failure or if alignment is not a
// power of two. The returned pointer must be released with aligned_free.
inline void* aligned_alloc_bytes(std::size_t bytes, std::size_t alignment) noexcept {
  if (!is_power_of_two(alignment)) {
    return nullptr;
  }
  if (bytes == 0) {
    return nullptr;
  }
  // posix_memalign requires alignment to be at least sizeof(void*) and a
  // power of two; round up to satisfy this.
  std::size_t actual_align = alignment < sizeof(void*) ? sizeof(void*) : alignment;
  void* p = nullptr;
  if (posix_memalign(&p, actual_align, bytes) != 0) {
    return nullptr;
  }
  return p;
}

inline void aligned_free(void* p) noexcept {
  // posix_memalign returns memory compatible with std::free.
  std::free(p);
}

// RAII container for an aligned typed buffer of trivially-copyable elements.
// AlignedBuffer<T> behaves like a fixed-size vector that does not initialise
// elements unless explicitly requested with zero(). For benchmarks this lets
// us measure raw kernel cost without paging-in time skewing results.
template <typename T>
class AlignedBuffer {
  static_assert(std::is_trivially_copyable<T>::value,
                "AlignedBuffer requires trivially copyable element type");

 public:
  AlignedBuffer() noexcept = default;

  AlignedBuffer(std::size_t count, std::size_t alignment = 64) noexcept
      : data_(static_cast<T*>(aligned_alloc_bytes(count * sizeof(T), alignment))),
        size_(data_ != nullptr ? count : 0),
        alignment_(alignment) {}

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.alignment_ = 0;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      size_ = other.size_;
      alignment_ = other.alignment_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.alignment_ = 0;
    }
    return *this;
  }

  ~AlignedBuffer() { release(); }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t alignment() const noexcept { return alignment_; }
  bool empty() const noexcept { return size_ == 0; }

  T& operator[](std::size_t i) noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }

  // Zero-fill the entire buffer.
  void zero() noexcept {
    if (data_ != nullptr) {
      std::memset(data_, 0, size_ * sizeof(T));
    }
  }

  // Detach the underlying pointer. Caller becomes responsible for calling
  // aligned_free.
  T* release_pointer() noexcept {
    T* p = data_;
    data_ = nullptr;
    size_ = 0;
    alignment_ = 0;
    return p;
  }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      aligned_free(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  T* data_{nullptr};
  std::size_t size_{0};
  std::size_t alignment_{0};
};

// Helper: returns true if the pointer is aligned to the given power-of-two
// boundary.
template <typename T>
inline bool is_aligned(const T* p, std::size_t alignment) noexcept {
  return is_power_of_two(alignment) &&
         (reinterpret_cast<std::uintptr_t>(p) & (alignment - 1)) == 0;
}

}  // namespace zhilicon::kernels::util
