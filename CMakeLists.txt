# SPDX-License-Identifier: Apache-2.0
#
# Repository root CMake entry point. This delegates to the C++ kernels
# project under benchmarks/. The harness Python tooling is built and
# packaged separately and does not require CMake.

cmake_minimum_required(VERSION 3.16)
project(zhilicon_benchmarks_root CXX)

enable_testing()
add_subdirectory(benchmarks)
