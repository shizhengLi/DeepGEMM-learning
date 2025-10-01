#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;
using namespace deep_gemm::sm100;

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
__global__ __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv, const uint64_t stride_kv,
                          uint32_t* cu_seq_len_k_start,
                          uint32_t* cu_seq_len_k_end,
                          float* logits,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // Normally, `h (kNumHeads) == 32` and `d (kHeadDim) == 64`
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    const auto& num_q_blocks = ceil_div(seq_len, BLOCK_Q);

    // Types
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warp_in_group_idx = warp_idx % 4;
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

    // Align to 512 bytes for swizzle-64B
    extern __shared__ __align__(512) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

    // TMA configs
    constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Data on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_kv_scales =  PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers     = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto full_umma_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups + i); });

    // Tensor memory allocation
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2);

    // Initialize barriers
    DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64, "Invalid threads");
    const bool& is_tma_load_warp = (warp_idx == (kNumMathThreads / 32));
    const bool& is_umma_warp = (warp_idx == (kNumMathThreads / 32 + 1));
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
            full_umma_barriers[i]->init(1);
            empty_umma_barriers[i]->init(128);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (is_umma_warp) {
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // Block scheduler
    uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
    const auto& get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + gridDim.x, q_iter_idx + 1};
    };
    const auto& load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto& q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            start = min(start, min(__ldg(cu_seq_len_k_start + q_idx), seq_len_kv));
            end = max(end, min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv));
        }
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, ceil_div(end - start, BLOCK_KV)};          // Task info
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto& get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    // UMMA settings
    // Construct instruction with layout F
    constexpr uint32_t UMMA_M = 64;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

    if (is_tma_load_warp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Prefetch
        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma_copy(&tensor_map_weights, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma_copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma_copy(&tensor_map_kv_scales, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else if (is_umma_warp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Issue UMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    empty_umma_barriers[i]->wait(((num_total_kv_blocks + kv_block_idx) & 1) ^ 1);
                    #pragma unroll
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                        auto a_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                        auto b_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_q[q_stage_idx], 0, k * UMMA_K);
                        cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                    }
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    } else if (warp_idx >= kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // Offsets
        const auto& tmem_start = __shfl_sync(0xffffffff, warpgroup_idx * UMMA_N, 0);
        float weights[BLOCK_Q][kNumHeads / 4];
        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                auto scale_kv = make_float2(ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset),
                                            ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset));

                // Wait UMMA arrival
                full_umma_barriers[warpgroup_idx]->wait((num_total_kv_blocks + kv_block_idx) & 1);

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Reduce over the head dim and store
                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Load from the tensor memory
                    constexpr uint32_t kNumLDTMElems = UMMA_M * kNumHeads / 128;
                    uint32_t shifted_accum[kNumLDTMElems];
                    DG_STATIC_ASSERT(kNumLDTMElems == 16 or kNumLDTMElems == 32 or kNumLDTMElems == 64, "Invalid LDTM");
                    auto tmem_load = [&](auto... Is) {
                        if constexpr (kNumLDTMElems == 16) {
                            cute::SM100_TMEM_LOAD_16dp256b4x::copy(tmem_start + i * kNumHeads, shifted_accum[Is]...);
                        } else if constexpr (kNumLDTMElems == 32) {
                            cute::SM100_TMEM_LOAD_16dp256b8x::copy(tmem_start + i * kNumHeads, shifted_accum[Is]...);
                        } else if constexpr (kNumLDTMElems == 64) {
                            cute::SM100_TMEM_LOAD_16dp256b16x::copy(tmem_start + i * kNumHeads, shifted_accum[Is]...);
                        }
                    };
                    [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load(Is...); }(cute::make_index_sequence<kNumLDTMElems>{});
                    cutlass::arch::fence_view_async_tmem_load();

                    // Release UMMA empty
                    if (i == BLOCK_Q - 1)
                        empty_umma_barriers[warpgroup_idx]->arrive();

                    // Transform
                    const auto& transform_2 = [&](const uint32_t& j, const uint32_t& k, const float2& sum) {
                        auto a = make_float2(fmaxf(*reinterpret_cast<float*>(&shifted_accum[j * 4 + k]), 0),
                                             fmaxf(*reinterpret_cast<float*>(&shifted_accum[j * 4 + k + 2]), 0));
                        auto b = make_float2(weights[i][j * 2 + k], weights[i][j * 2 + k]);
                        return __ffma2_rn(a, b, sum);
                    };

                    // Intra-thread reduction
                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 8; ++ j) {
                        sum_0 = transform_2(j, 0, sum_0);
                        sum_1 = transform_2(j, 1, sum_1);
                    }
                    auto v = __fmul2_rn(__fadd2_rn(sum_0, sum_1), scale_kv);

                    // Inter-thread reduction
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = 1u << j;
                        v.x += __shfl_xor_sync(0xffffffffu, v.x, offset);
                        v.y += __shfl_xor_sync(0xffffffffu, v.y, offset);
                    }

                    // Store into the global memory
                    // NOTES: we have redundant writes here, consider more carefully
                    const uint32_t& q_idx = block_q_idx * BLOCK_Q + i;
                    logits[q_idx * stride_kv + kv_offset + v_0_offset] = v.x;
                    logits[q_idx * stride_kv + kv_offset + v_1_offset] = v.y;
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }

    // Free tensor memory
    __syncthreads();
    if (is_tma_load_warp)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
}

} // namespace deep_gemm
