// Copyright 2026 RISC-V RVV optimization work.
//
// kernel_rvv.cc: RVV 1.0 8x16 depth-4 GEMM kernel (uint8 inputs, int32
// accumulators). Compiled with -march=rv64gcv; see kernel_rvv.h.
//
// Layout notes (matches gemmlowp packing for the format declared in
// kernel_rvv.h):
//   LHS: one 8x4 DepthMajor cell per kernel row-block. For depth d, the
//        8 row values are contiguous at lhs + d*8 -- one vle8 feeds all
//        16 columns.
//   RHS: four 4x4 DepthMajor cells (16 columns). Element (c, d) of cell
//        X is at rhs + X*16 + c + d*4.
//   DST: column-major packed result, dst_row_stride == 1. Column c lives
//        at dst + c * dst_col_stride and holds 8 contiguous int32, so
//        accumulator vectors load/store contiguously.
//
// MAC chain: per depth step, one widened LHS vector (8 lanes = rows) is
// consumed by 16 vwmacc.vq (scalar RHS value x vector LHS), i.e. one
// vector op per 8 MACs, amortizing the LHS load across all 16 columns.

#include "kernel_rvv.h"

#include <riscv_vector.h>

#include <cassert>
#include <cstdint>

namespace gemmlowp {

const char* RVV_Kernel8x16Depth4::Name() const {
  return "RVV, 8x16, depth 4";
}

#define GEMMLOWP_RVV_MAC(ACC, VAL) \
  (ACC) = __riscv_vwmacc_vx_i32m1((ACC), (std::int16_t)(VAL), lv, 8)

void RVV_Kernel8x16Depth4::Run(std::int32_t* dst_ptr,
                               std::size_t dst_row_stride,
                               std::size_t dst_col_stride,
                               const std::uint8_t* lhs_ptr,
                               const std::uint8_t* rhs_ptr,
                               std::size_t start_depth,
                               std::size_t run_depth) const {
  (void)dst_row_stride;
  assert(dst_row_stride == 1);

  constexpr std::size_t kRows = 8;
  constexpr std::size_t kCols = 16;
  constexpr std::size_t kDepth = 4;
  constexpr std::size_t kVL = 8;  // kRows int32 lanes

  // 16 accumulators, one per output column (X = rhs cell, Y = column in
  // cell), each holding the 8 output rows.
  vint32m1_t c00, c01, c02, c03, c10, c11, c12, c13;
  vint32m1_t c20, c21, c22, c23, c30, c31, c32, c33;

  if (start_depth == 0) {
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, kVL);
    c00 = c01 = c02 = c03 = zero;
    c10 = c11 = c12 = c13 = zero;
    c20 = c21 = c22 = c23 = zero;
    c30 = c31 = c32 = c33 = zero;
  } else {
    // Continue accumulating into the existing packed-result block.
    std::int32_t* d00 = dst_ptr;
    std::int32_t* d1 = dst_ptr + 4 * dst_col_stride;
    std::int32_t* d2 = dst_ptr + 8 * dst_col_stride;
    std::int32_t* d3 = dst_ptr + 12 * dst_col_stride;
    c00 = __riscv_vle32_v_i32m1(d00, kVL);
    c01 = __riscv_vle32_v_i32m1(d00 + dst_col_stride, kVL);
    c02 = __riscv_vle32_v_i32m1(d00 + 2 * dst_col_stride, kVL);
    c03 = __riscv_vle32_v_i32m1(d00 + 3 * dst_col_stride, kVL);
    c10 = __riscv_vle32_v_i32m1(d1, kVL);
    c11 = __riscv_vle32_v_i32m1(d1 + dst_col_stride, kVL);
    c12 = __riscv_vle32_v_i32m1(d1 + 2 * dst_col_stride, kVL);
    c13 = __riscv_vle32_v_i32m1(d1 + 3 * dst_col_stride, kVL);
    c20 = __riscv_vle32_v_i32m1(d2, kVL);
    c21 = __riscv_vle32_v_i32m1(d2 + dst_col_stride, kVL);
    c22 = __riscv_vle32_v_i32m1(d2 + 2 * dst_col_stride, kVL);
    c23 = __riscv_vle32_v_i32m1(d2 + 3 * dst_col_stride, kVL);
    c30 = __riscv_vle32_v_i32m1(d3, kVL);
    c31 = __riscv_vle32_v_i32m1(d3 + dst_col_stride, kVL);
    c32 = __riscv_vle32_v_i32m1(d3 + 2 * dst_col_stride, kVL);
    c33 = __riscv_vle32_v_i32m1(d3 + 3 * dst_col_stride, kVL);
  }

  const std::size_t depth_cells = run_depth / kDepth;
  for (std::size_t dc = 0; dc < depth_cells; ++dc) {
    const std::uint8_t* lc = lhs_ptr + dc * (kRows * kDepth);
    const std::uint8_t* rc = rhs_ptr + dc * (kCols * kDepth);
    for (std::size_t d = 0; d < kDepth; ++d) {
      const vuint8mf4_t lb = __riscv_vle8_v_u8mf4(lc + d * kRows, kVL);
      const vint16mf2_t lv = __riscv_vreinterpret_v_u16mf2_i16mf2(
          __riscv_vwcvtu_x_x_v_u16mf2(lb, kVL));
      // Four RHS cells, 4 columns each; scalar RHS value x vector LHS.
      const std::uint8_t* r0 = rc + d * 4;
      const std::uint8_t* r1 = r0 + kCols;
      const std::uint8_t* r2 = r0 + 2 * kCols;
      const std::uint8_t* r3 = r0 + 3 * kCols;
      GEMMLOWP_RVV_MAC(c00, r0[0]);
      GEMMLOWP_RVV_MAC(c01, r0[1]);
      GEMMLOWP_RVV_MAC(c02, r0[2]);
      GEMMLOWP_RVV_MAC(c03, r0[3]);
      GEMMLOWP_RVV_MAC(c10, r1[0]);
      GEMMLOWP_RVV_MAC(c11, r1[1]);
      GEMMLOWP_RVV_MAC(c12, r1[2]);
      GEMMLOWP_RVV_MAC(c13, r1[3]);
      GEMMLOWP_RVV_MAC(c20, r2[0]);
      GEMMLOWP_RVV_MAC(c21, r2[1]);
      GEMMLOWP_RVV_MAC(c22, r2[2]);
      GEMMLOWP_RVV_MAC(c23, r2[3]);
      GEMMLOWP_RVV_MAC(c30, r3[0]);
      GEMMLOWP_RVV_MAC(c31, r3[1]);
      GEMMLOWP_RVV_MAC(c32, r3[2]);
      GEMMLOWP_RVV_MAC(c33, r3[3]);
    }
  }

  std::int32_t* o00 = dst_ptr;
  std::int32_t* o1 = dst_ptr + 4 * dst_col_stride;
  std::int32_t* o2 = dst_ptr + 8 * dst_col_stride;
  std::int32_t* o3 = dst_ptr + 12 * dst_col_stride;
  __riscv_vse32_v_i32m1(o00, c00, kVL);
  __riscv_vse32_v_i32m1(o00 + dst_col_stride, c01, kVL);
  __riscv_vse32_v_i32m1(o00 + 2 * dst_col_stride, c02, kVL);
  __riscv_vse32_v_i32m1(o00 + 3 * dst_col_stride, c03, kVL);
  __riscv_vse32_v_i32m1(o1, c10, kVL);
  __riscv_vse32_v_i32m1(o1 + dst_col_stride, c11, kVL);
  __riscv_vse32_v_i32m1(o1 + 2 * dst_col_stride, c12, kVL);
  __riscv_vse32_v_i32m1(o1 + 3 * dst_col_stride, c13, kVL);
  __riscv_vse32_v_i32m1(o2, c20, kVL);
  __riscv_vse32_v_i32m1(o2 + dst_col_stride, c21, kVL);
  __riscv_vse32_v_i32m1(o2 + 2 * dst_col_stride, c22, kVL);
  __riscv_vse32_v_i32m1(o2 + 3 * dst_col_stride, c23, kVL);
  __riscv_vse32_v_i32m1(o3, c30, kVL);
  __riscv_vse32_v_i32m1(o3 + dst_col_stride, c31, kVL);
  __riscv_vse32_v_i32m1(o3 + 2 * dst_col_stride, c32, kVL);
  __riscv_vse32_v_i32m1(o3 + 3 * dst_col_stride, c33, kVL);
}

#undef GEMMLOWP_RVV_MAC

}  // namespace gemmlowp
