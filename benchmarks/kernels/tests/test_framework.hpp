// SPDX-License-Identifier: Apache-2.0
// zhilicon-benchmarks: tiny test framework.
//
// We avoid pulling in gtest as a hard dependency so that CI runners
// without internet (FetchContent) and without a system-wide gtest install
// can still validate the kernels. The framework is intentionally minimal:
// macros register tests at static-init time and the main() in each
// _test.cpp iterates and reports.
//
// Each test is a function with signature void(TestContext&) where the
// context tracks pass/fail counts. CHECK / CHECK_EQ / CHECK_NEAR record
// a failure and continue, so a single test can report multiple findings.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace zhilicon::kernels::testing {

struct TestContext {
  int checks{0};
  int failures{0};
  const char* current_test{""};
  std::string failure_log;

  void record_failure(const char* file, int line, const std::string& msg) {
    ++failures;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s:%d", file, line);
    failure_log += "  [FAIL] ";
    failure_log += current_test;
    failure_log += " @ ";
    failure_log += buf;
    failure_log += " : ";
    failure_log += msg;
    failure_log += '\n';
  }
};

struct TestEntry {
  const char* name;
  std::function<void(TestContext&)> fn;
};

inline std::vector<TestEntry>& registry() {
  static std::vector<TestEntry> r;
  return r;
}

struct TestRegistrar {
  TestRegistrar(const char* name, std::function<void(TestContext&)> fn) {
    registry().push_back({name, std::move(fn)});
  }
};

inline int run_all(const char* suite_name) {
  TestContext ctx;
  int run = 0;
  int failed = 0;
  for (const auto& entry : registry()) {
    ctx.current_test = entry.name;
    int prev_failures = ctx.failures;
    entry.fn(ctx);
    ++run;
    if (ctx.failures > prev_failures) {
      ++failed;
    }
  }
  std::printf("[%s] %d tests, %d checks, %d test failures\n", suite_name, run,
              ctx.checks, failed);
  if (!ctx.failure_log.empty()) {
    std::printf("%s", ctx.failure_log.c_str());
  }
  return failed == 0 ? 0 : 1;
}

}  // namespace zhilicon::kernels::testing

#define ZH_TEST(name)                                                        \
  static void zh_test_##name(                                                \
      ::zhilicon::kernels::testing::TestContext& ctx_);                      \
  static ::zhilicon::kernels::testing::TestRegistrar zh_reg_##name(          \
      #name, zh_test_##name);                                                \
  static void zh_test_##name(                                                \
      ::zhilicon::kernels::testing::TestContext& ctx_)

#define ZH_CHECK(cond)                                                       \
  do {                                                                       \
    ++ctx_.checks;                                                           \
    if (!(cond)) {                                                           \
      ctx_.record_failure(__FILE__, __LINE__, #cond);                        \
    }                                                                        \
  } while (0)

#define ZH_CHECK_EQ(a, b)                                                    \
  do {                                                                       \
    ++ctx_.checks;                                                           \
    auto _zh_eq_a = (a);                                                     \
    auto _zh_eq_b = (b);                                                     \
    if (!(_zh_eq_a == _zh_eq_b)) {                                           \
      ctx_.record_failure(__FILE__, __LINE__,                                \
                          std::string(#a " != " #b));                        \
    }                                                                        \
  } while (0)

#define ZH_CHECK_NEAR(a, b, tol)                                             \
  do {                                                                       \
    ++ctx_.checks;                                                           \
    double _zh_a = static_cast<double>(a);                                   \
    double _zh_b = static_cast<double>(b);                                   \
    double _zh_t = static_cast<double>(tol);                                 \
    if (!(std::fabs(_zh_a - _zh_b) <= _zh_t)) {                              \
      char _zh_buf[128];                                                     \
      std::snprintf(_zh_buf, sizeof(_zh_buf),                                \
                    "%s (%.10g) and %s (%.10g) differ by %.10g > %.10g",     \
                    #a, _zh_a, #b, _zh_b, std::fabs(_zh_a - _zh_b), _zh_t);  \
      ctx_.record_failure(__FILE__, __LINE__, _zh_buf);                      \
    }                                                                        \
  } while (0)

#define ZH_TEST_MAIN(suite)                                                  \
  int main() {                                                               \
    return ::zhilicon::kernels::testing::run_all(suite);                     \
  }
