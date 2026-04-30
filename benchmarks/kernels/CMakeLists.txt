# SPDX-License-Identifier: Apache-2.0
#
# zhilicon-benchmarks: build configuration for the C++17 kernel library.
#
# The kernels are header-only; this CMakeLists.txt builds the unit tests
# that exercise each header. Each test binary registers with ctest so
# `ctest --output-on-failure` runs the full suite.

cmake_minimum_required(VERSION 3.16)
project(zhilicon_benchmark_kernels CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Header-only kernel interface library so consumers can link against it
# without having to wire up include paths manually.
add_library(zhilicon_kernels INTERFACE)
target_include_directories(zhilicon_kernels INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR})

# Common compile options. -Werror keeps the suite honest.
set(ZH_KERNEL_WARNINGS
  -Wall
  -Wextra
  -Werror
  -Wpedantic
  -Wshadow
  -Wconversion
  -Wsign-conversion
  -Wnon-virtual-dtor
  -Wnull-dereference
  -Wdouble-promotion
  -Wformat=2
  -Wno-unused-parameter)

enable_testing()

# Helper: register a single test binary from a single source file.
function(zh_add_kernel_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE zhilicon_kernels)
  target_compile_options(${name} PRIVATE ${ZH_KERNEL_WARNINGS})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

zh_add_kernel_test(aligned_test       tests/aligned_test.cpp)
zh_add_kernel_test(bit_test           tests/bit_test.cpp)
zh_add_kernel_test(half_test          tests/half_test.cpp)
zh_add_kernel_test(pseudorandom_test  tests/pseudorandom_test.cpp)
zh_add_kernel_test(fir_test           tests/fir_test.cpp)
zh_add_kernel_test(iir_test           tests/iir_test.cpp)
zh_add_kernel_test(fft_test           tests/fft_test.cpp)
zh_add_kernel_test(window_test        tests/window_test.cpp)
zh_add_kernel_test(matmul_test        tests/matmul_test.cpp)
zh_add_kernel_test(transpose_test     tests/transpose_test.cpp)
zh_add_kernel_test(qr_test            tests/qr_test.cpp)
zh_add_kernel_test(quantize_test      tests/quantize_test.cpp)
zh_add_kernel_test(int8_mm_test       tests/int8_mm_test.cpp)
zh_add_kernel_test(dequant_mm_test    tests/dequant_mm_test.cpp)
zh_add_kernel_test(parallel_sum_test  tests/parallel_sum_test.cpp)
zh_add_kernel_test(variance_test      tests/variance_test.cpp)
zh_add_kernel_test(argmax_test        tests/argmax_test.cpp)
zh_add_kernel_test(im2col_test        tests/im2col_test.cpp)
zh_add_kernel_test(conv2d_test        tests/conv2d_test.cpp)
