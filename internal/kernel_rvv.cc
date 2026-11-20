// Copyright 2026 RISC-V RVV optimization work.
//
// kernel_rvv.cc: RVV 1.0 16x8 depth-4 GEMM kernel (uint8 inputs, int32
// accumulators). Compiled with -march=rv64gcv; see kernel_rvv.h.
//
//   LHS: one 16x4 DepthMajor cell. For depth d, 16 row values are contiguous
//        at lhs + d*16 -- one vle8 (u8mf2, 16 lanes) feeds all 8 columns.
//   RHS: two 4x4 DepthMajor cells (8 columns). Element (c, d) of cell X is
//        at rhs + X*16 + c + d*4.
//   DST: column-major packed result; column c holds 16 contiguous int32,
//        i.e. one vint32m2_t, so accumulators load/store contiguously.

#include "kernel_rvv.h"

#include <riscv_vector.h>

#include <cassert>
#include <cstdint>

namespace gemmlowp {

const char* RVV_Kernel16x8Depth4::Name() const {
  return "RVV, 16x8, depth 4";
}

#define GEMMLOWP_RVV_MAC(ACC, VAL) \
  (ACC) = __riscv_vwmacc_vx_i32m2((ACC), (std::int16_t)(VAL), lv, 16)

void RVV_Kernel16x8Depth4::Run(std::int32_t* dst_ptr,
                               std::size_t dst_row_stride,
                               std::size_t dst_col_stride,
                               const std::uint8_t* lhs_ptr,
                               const std::uint8_t* rhs_ptr,
                               std::size_t start_depth,
                               std::size_t run_depth) const {
  (void)dst_row_stride;
  assert(dst_row_stride == 1);

  constexpr std::size_t kRows = 16;
  constexpr std::size_t kCols = 8;
  constexpr std::size_t kDepth = 4;
  constexpr std::size_t kVL = 16;  // 16 rows per i32m2 accumulator

  vint32m2_t c00, c01, c02, c03, c10, c11, c12, c13;

  if (start_depth == 0) {
    const vint32m2_t zero = __riscv_vmv_v_x_i32m2(0, kVL);
    c00 = c01 = c02 = c03 = zero;
    c10 = c11 = c12 = c13 = zero;
  } else {
    c00 = __riscv_vle32_v_i32m2(dst_ptr, kVL);
    c01 = __riscv_vle32_v_i32m2(dst_ptr + dst_col_stride, kVL);
    c02 = __riscv_vle32_v_i32m2(dst_ptr + 2 * dst_col_stride, kVL);
    c03 = __riscv_vle32_v_i32m2(dst_ptr + 3 * dst_col_stride, kVL);
    c10 = __riscv_vle32_v_i32m2(dst_ptr + 4 * dst_col_stride, kVL);
    c11 = __riscv_vle32_v_i32m2(dst_ptr + 5 * dst_col_stride, kVL);
    c12 = __riscv_vle32_v_i32m2(dst_ptr + 6 * dst_col_stride, kVL);
    c13 = __riscv_vle32_v_i32m2(dst_ptr + 7 * dst_col_stride, kVL);
  }

  const std::size_t depth_cells = run_depth / kDepth;
  for (std::size_t dc = 0; dc < depth_cells; ++dc) {
    const std::uint8_t* lc = lhs_ptr + dc * (kRows * kDepth);
    const std::uint8_t* rc = rhs_ptr + dc * (kCols * kDepth);
    for (std::size_t d = 0; d < kDepth; ++d) {
      const vuint8mf2_t lb = __riscv_vle8_v_u8mf2(lc + d * kRows, kVL);
      const vint16m1_t lv = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(lb, kVL));
      const std::uint8_t* r0 = rc + d * 4;
      const std::uint8_t* r1 = r0 + 16;
      GEMMLOWP_RVV_MAC(c00, r0[0]);
      GEMMLOWP_RVV_MAC(c01, r0[1]);
      GEMMLOWP_RVV_MAC(c02, r0[2]);
      GEMMLOWP_RVV_MAC(c03, r0[3]);
      GEMMLOWP_RVV_MAC(c10, r1[0]);
      GEMMLOWP_RVV_MAC(c11, r1[1]);
      GEMMLOWP_RVV_MAC(c12, r1[2]);
      GEMMLOWP_RVV_MAC(c13, r1[3]);
    }
  }

  __riscv_vse32_v_i32m2(dst_ptr, c00, kVL);
  __riscv_vse32_v_i32m2(dst_ptr + dst_col_stride, c01, kVL);
  __riscv_vse32_v_i32m2(dst_ptr + 2 * dst_col_stride, c02, kVL);
  __riscv_vse32_v_i32m2(dst_ptr + 3 * dst_col_stride, c03, kVL);
  __riscv_vse32_v_i32m2(dst_ptr + 4 * dst_col_stride, c10, kVL);
  __riscv_vse32_v_i32m2(dst_ptr + 5 * dst_col_stride, c11, kVL);
  __riscv_vse32_v_i32m2(dst_ptr + 6 * dst_col_stride, c12, kVL);
  __riscv_vse32_v_i32m2(dst_ptr + 7 * dst_col_stride, c13, kVL);
}

#undef GEMMLOWP_RVV_MAC

}  // namespace gemmlowp
