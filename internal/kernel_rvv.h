// Copyright 2026 RISC-V RVV optimization work.
//
// kernel_rvv.h: RVV 1.0 GEMM kernel for RISC-V (SpacemiT K3 class cores,
// VLEN=256). The implementation lives in kernel_rvv.cc, which is compiled
// with -march=rv64gcv; the rest of gemmlowp stays scalar.

#ifndef GEMMLOWP_INTERNAL_KERNEL_RVV_H_
#define GEMMLOWP_INTERNAL_KERNEL_RVV_H_

#include "kernel.h"

namespace gemmlowp {

// 16 rows x 8 columns, depth 4. i32m2 accumulators (16 lanes) let each
// widening MAC cover all 16 output rows, halving the number of scalar RHS
// loads and DST vector stores versus the 8x16 i32m1 layout.
class RVV_Kernel16x8Depth4 : public KernelBase {
 public:
  typedef KernelFormat<KernelSideFormat<CellFormat<16, 4>, 1>,
                       KernelSideFormat<CellFormat<4, 4>, 2> >
      Format;

  const char* Name() const override;
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override;
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_RVV_H_
