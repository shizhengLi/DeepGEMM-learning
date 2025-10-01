// GEMM kernels
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm100_fp8_gemm_1d2d.cuh>

// Attention kernels
#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>
#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>
#include <deep_gemm/impls/sm100_fp8_mqa_logits.cuh>
#include <deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh>
#include <deep_gemm/impls/smxx_clean_logits.cuh>

// Einsum kernels
#include <deep_gemm/impls/sm90_bmk_bnk_mn.cuh>
#include <deep_gemm/impls/sm100_bmk_bnk_mn.cuh>

// Layout kernels
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

int main() {
    return 0;
}
