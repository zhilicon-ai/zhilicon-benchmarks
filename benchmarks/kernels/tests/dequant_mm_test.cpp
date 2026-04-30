// SPDX-License-Identifier: Apache-2.0
#include "test_framework.hpp"
#include "../quant/dequant_mm.hpp"
#include "../quant/int8_mm.hpp"

#include <cmath>
#include <vector>

using namespace zhilicon::kernels::quant;

ZH_TEST(dequant_matmul_zero_zero_points_matches_int_matmul_scaled) {
  std::vector<std::int8_t> A = {1, 2, 3, 4};   // 2x2
  std::vector<std::int8_t> B = {1, 0, 0, 1};   // 2x2 identity
  std::vector<float> C(4, 0.0f);
  QuantParams pA;
  pA.scale = 0.5f;
  QuantParams pB;
  pB.scale = 2.0f;
  dequant_matmul(A.data(), 2, pA, B.data(), 2, pB, C.data(), 2, 2, 2, 2);
  // result = (A_int * B_int) * 0.5 * 2.0 = A_int (since B is identity)
  ZH_CHECK_NEAR(C[0], 1.0f, 1e-6);
  ZH_CHECK_NEAR(C[1], 2.0f, 1e-6);
  ZH_CHECK_NEAR(C[2], 3.0f, 1e-6);
  ZH_CHECK_NEAR(C[3], 4.0f, 1e-6);
}

ZH_TEST(dequant_matmul_with_zero_points) {
  std::vector<std::int8_t> A = {10, 12, 14, 16};  // 2x2
  std::vector<std::int8_t> B = {1, 0, 0, 1};
  std::vector<float> C(4, 0.0f);
  QuantParams pA;
  pA.scale = 0.25f;
  pA.zero_point = 10;       // subtracts 10 from all A entries
  QuantParams pB;
  pB.scale = 1.0f;
  pB.zero_point = 0;
  dequant_matmul(A.data(), 2, pA, B.data(), 2, pB, C.data(), 2, 2, 2, 2);
  // (A - 10) = [0, 2; 4, 6], * I * 0.25 * 1.0
  ZH_CHECK_NEAR(C[0], 0.0f, 1e-6);
  ZH_CHECK_NEAR(C[1], 0.5f, 1e-6);
  ZH_CHECK_NEAR(C[2], 1.0f, 1e-6);
  ZH_CHECK_NEAR(C[3], 1.5f, 1e-6);
}

ZH_TEST(dequant_matmul_zero_input_gives_zero) {
  std::vector<std::int8_t> A(16, 0);
  std::vector<std::int8_t> B(16, 0);
  std::vector<float> C(16, 99.0f);
  QuantParams pA;
  pA.scale = 1.0f;
  QuantParams pB;
  pB.scale = 1.0f;
  dequant_matmul(A.data(), 4, pA, B.data(), 4, pB, C.data(), 4, 4, 4, 4);
  for (auto v : C) {
    ZH_CHECK_NEAR(v, 0.0f, 0.0f);
  }
}

ZH_TEST(dequant_matmul_per_row_col_known_values) {
  std::vector<std::int8_t> A = {1, 2, 3, 4};
  std::vector<std::int8_t> B = {1, 0, 0, 1};
  std::vector<float> rs = {2.0f, 3.0f};
  std::vector<float> cs = {4.0f, 5.0f};
  std::vector<float> C(4, 0.0f);
  dequant_matmul_per_row_col(A.data(), 2, rs.data(), B.data(), 2, cs.data(),
                             C.data(), 2, 2, 2, 2);
  // (A * I) = A, then * row_scale[i] * col_scale[j].
  // C[0,0] = 1 * 2 * 4 = 8
  // C[0,1] = 2 * 2 * 5 = 20
  // C[1,0] = 3 * 3 * 4 = 36
  // C[1,1] = 4 * 3 * 5 = 60
  ZH_CHECK_NEAR(C[0], 8.0f, 1e-6);
  ZH_CHECK_NEAR(C[1], 20.0f, 1e-6);
  ZH_CHECK_NEAR(C[2], 36.0f, 1e-6);
  ZH_CHECK_NEAR(C[3], 60.0f, 1e-6);
}

ZH_TEST(dequant_matmul_consistency_with_int_then_scale) {
  // Verify that fused dequant_matmul agrees with explicit int8 mm + scale.
  std::vector<std::int8_t> A = {-1, 2, -3, 4, 5, -6};   // 2x3
  std::vector<std::int8_t> B = {1, 0, -1, 2, 3, -1};    // 3x2
  std::vector<std::int32_t> C_int(4, 0);
  int8_matmul(A.data(), 3, B.data(), 2, C_int.data(), 2, 2, 2, 3);

  QuantParams pA;
  pA.scale = 0.1f;
  QuantParams pB;
  pB.scale = 0.2f;
  std::vector<float> C_fused(4, 0.0f);
  dequant_matmul(A.data(), 3, pA, B.data(), 2, pB, C_fused.data(), 2, 2, 2, 3);
  for (std::size_t i = 0; i < 4; ++i) {
    float expected = static_cast<float>(C_int[i]) * pA.scale * pB.scale;
    ZH_CHECK_NEAR(C_fused[i], expected, 1e-5);
  }
}

ZH_TEST(dequant_matmul_negative_zero_point) {
  std::vector<std::int8_t> A = {0, 0, 0, 0};
  std::vector<std::int8_t> B = {1, 0, 0, 1};
  std::vector<float> C(4, 0.0f);
  QuantParams pA;
  pA.scale = 1.0f;
  pA.zero_point = -5;
  QuantParams pB;
  pB.scale = 1.0f;
  pB.zero_point = 0;
  dequant_matmul(A.data(), 2, pA, B.data(), 2, pB, C.data(), 2, 2, 2, 2);
  // (A - -5) = [5,5;5,5], * I = [5,5;5,5]
  for (auto v : C) {
    ZH_CHECK_NEAR(v, 5.0f, 1e-6);
  }
}

ZH_TEST(dequant_matmul_per_row_col_zero_scales) {
  // Zero scaling should produce zero output.
  std::vector<std::int8_t> A = {1, 2, 3, 4};
  std::vector<std::int8_t> B = {1, 1, 1, 1};
  std::vector<float> rs = {0.0f, 0.0f};
  std::vector<float> cs = {1.0f, 1.0f};
  std::vector<float> C(4, 0.0f);
  dequant_matmul_per_row_col(A.data(), 2, rs.data(), B.data(), 2, cs.data(),
                             C.data(), 2, 2, 2, 2);
  for (auto v : C) {
    ZH_CHECK_NEAR(v, 0.0f, 0.0f);
  }
}

ZH_TEST_MAIN("quant/dequant_mm")
