// Copyright 2026 RISC-V RVV optimization work.
//
// output_rvv.h: RVV 1.0 specialization of the standard output pipeline
// (QuantizeDownInt32ToUint8Scale + SaturatingCastToUint8). It bypasses the
// generic simd_wrappers scalar path by operating directly on the contiguous
// register block held on the stack, bridging scalar reg[] arrays into RVV
// vectors locally (RVV sizeless types cannot be struct members).

#ifndef GEMMLOWP_INTERNAL_OUTPUT_RVV_H_
#define GEMMLOWP_INTERNAL_OUTPUT_RVV_H_

#include <riscv_vector.h>

#include <cstdint>
#include <tuple>

namespace gemmlowp {

// Requantize + saturate a contiguous run of `n` int32 values into uint8.
// Bit-exact match of OutputStageQuantizeDownInt32ToUint8Scale +
// OutputStageSaturatingCastToUint8 (gemmlowp RoundingDivideByPOT semantics).
inline void RvvRequantizeSaturateU8(const std::int32_t* src, std::uint8_t* dst,
                                    int n, std::int32_t result_offset,
                                    std::int32_t result_mult, int result_shift) {
  const int mask = (int)((1LL << result_shift) - 1);
  const int half = mask >> 1;
  int processed = 0;
  while (processed < n) {
    const int vl = n - processed > 8 ? 8 : n - processed;
    const std::int32_t* p = src + processed;
    vint32m1_t x = __riscv_vle32_v_i32m1(p, vl);
    x = __riscv_vadd_vx_i32m1(x, result_offset, vl);
    x = __riscv_vmul_vx_i32m1(x, result_mult, vl);
    // RoundingDivideByPOT
    vint32m1_t rem = __riscv_vand_vx_i32m1(x, mask, vl);
    vbool32_t is_neg = __riscv_vmslt_vx_i32m1_b32(x, 0, vl);
    vint32m1_t threshold = __riscv_vadd_vv_i32m1(
        __riscv_vmv_v_x_i32m1(half, vl),
        __riscv_vmerge_vxm_i32m1(__riscv_vmv_v_x_i32m1(0, vl), 1, is_neg, vl),
        vl);
    vbool32_t gt = __riscv_vmsgtu_vv_u32m1_b32(
        __riscv_vreinterpret_v_i32m1_u32m1(rem),
        __riscv_vreinterpret_v_i32m1_u32m1(threshold), vl);
    vint32m1_t rounded = __riscv_vadd_vv_i32m1(
        __riscv_vsra_vx_i32m1(x, result_shift, vl),
        __riscv_vmerge_vxm_i32m1(__riscv_vmv_v_x_i32m1(0, vl), 1, gt, vl), vl);
    vint32m1_t clamped = __riscv_vmax_vx_i32m1(
        __riscv_vmin_vx_i32m1(rounded, 255, vl), 0, vl);
    vuint16mf2_t mid = __riscv_vnclipu_wx_u16mf2(
        __riscv_vreinterpret_v_i32m1_u32m1(clamped), 0, __RISCV_VXRM_RDN, vl);
    vuint8mf4_t b = __riscv_vnclipu_wx_u8mf4(mid, 0, __RISCV_VXRM_RDN, vl);
    __riscv_vse8_v_u8mf4(dst + processed, b, vl);
    processed += vl;
  }
}

// RVV specialization of the executor for the standard u8 output pipeline.
// InputType is RegisterBlock<int32_t, Rows, Cols> produced by unpack.
template <typename InputType>
struct OutputPipelineExecutor<
    std::tuple<OutputStageQuantizeDownInt32ToUint8Scale,
               OutputStageSaturatingCastToUint8>,
    InputType> {
  typedef std::tuple<OutputStageQuantizeDownInt32ToUint8Scale,
                     OutputStageSaturatingCastToUint8> Pipeline;
  static constexpr int Rows = InputType::kRows;
  static constexpr int Cols = InputType::kCols;

  OutputPipelineExecutor(const Pipeline& output_pipeline)
      : result_offset_(std::get<0>(output_pipeline).result_offset),
        result_mult_(std::get<0>(output_pipeline).result_mult_int),
        result_shift_(std::get<0>(output_pipeline).result_shift) {}

  template <typename DstType>
  void Execute(InputType input, DstType* dst, int, int, int dst_row,
               int dst_col) const {
    // The register block's buf.reg[] is a contiguous col-major int32 array:
    // element (r, c) lives at reg[r + c * Rows].
    std::int32_t acc[Rows * Cols];
    for (int i = 0; i < Rows * Cols; i++) acc[i] = input.buf.reg[i];
    std::uint8_t out[Rows * Cols];
    RvvRequantizeSaturateU8(acc, out, Rows * Cols, result_offset_,
                            result_mult_, result_shift_);
    // Store back using the same layout as generic StoreFinalOutput:
    // dst(r, c) = reg[r + c * Rows].
    for (int c = 0; c < Cols; c++) {
      for (int r = 0; r < Rows; r++) {
        *dst->data(dst_row + r, dst_col + c) = out[r + c * Rows];
      }
    }
  }

  std::int32_t result_offset_;
  std::int32_t result_mult_;
  int result_shift_;
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_RVV_H_
