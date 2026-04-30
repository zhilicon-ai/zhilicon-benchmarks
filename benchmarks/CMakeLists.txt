# SPDX-License-Identifier: Apache-2.0
#
# Top-level CMakeLists for the zhilicon-benchmarks C++ benchmark code.
# The Python harness scripts in tools/ remain the primary entry point;
# this CMake project builds the supporting C++ kernel library that the
# harness loads via shared objects (or, for the test path, runs as
# stand-alone executables under ctest).

cmake_minimum_required(VERSION 3.16)
project(zhilicon_benchmarks CXX)

add_subdirectory(kernels)
