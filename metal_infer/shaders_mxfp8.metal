/*
 * shaders_mxfp8.metal — Metal compute shader for MXFP8 (E4M3 weight + E8M0 scale)
 *
 * MXFP8 format (Microscaling):
 *   - Each weight is FP8 E4M3 (1 sign + 4 exponent + 3 mantissa) stored as uint8
 *   - 4 values packed per uint32
 *   - Per-group scale is E8M0 (1 sign + 7 exponent, pure power of 2) stored as uint8
 *   - group_size = 32 for all weights
 *
 * Dequantization:
 *   value_fp32 = e4m3_to_fp32(weight_byte) * e8m0_to_fp32(scale_byte)
 *
 * E8M0 format: sign(1) + exponent(7), value = (-1)^sign * 2^(exp - 127)
 *   Special: 0xFF = NaN, 0x7F = 1.0
 *
 * Author: Phoenix (for Tuan Andre / Panglima Ekspres)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// FP8 E4M3 to FP32 conversion
// ============================================================================
static inline float e4m3_to_f32(uint8_t v) {
    uint sign     = (v >> 7) & 1;
    uint exponent = (v >> 3) & 0xF;
    uint mantissa = v & 0x7;

    if (exponent == 0 && mantissa == 0) return 0.0f;

    // NaN: exp=15, mantissa=7
    if (exponent == 15 && mantissa == 7) {
        return sign ? as_type<float>(0xFFC00000u) : as_type<float>(0x7FC00000u);
    }

    if (exponent == 0) {
        // Subnormal E4M3
        uint32_t bits = (sign << 31) | (120u << 23) | (mantissa << 20);
        return as_type<float>(bits) * 0.5f;
    }

    // Normal
    int fp32_exp = (int)exponent - 7 + 127;
    uint32_t bits = (sign << 31) | ((uint32_t)fp32_exp << 23) | (mantissa << 20);
    return as_type<float>(bits);
}

// ============================================================================
// E8M0 to FP32 conversion (pure power-of-2 scale factor)
// ============================================================================
// E8M0: sign(1) + exponent(7), no mantissa
// value = (-1)^sign * 2^(exp - 127)
// Special: 0xFF = NaN
static inline float e8m0_to_f32(uint8_t b) {
    if (b == 0xFF) return as_type<float>(0x7FC00000u); // NaN

    uint sign = (b >> 7) & 1;
    uint exp7 = b & 0x7F;

    // Build FP32: 2^(exp7 - 127)
    // FP32 bias = 127, so FP32 exponent = exp7
    uint32_t bits = (sign << 31) | (exp7 << 23);
    return as_type<float>(bits);
}

// ============================================================================
// Kernel: MXFP8 dequant matvec (E4M3 weight + E8M0 scale)
// ============================================================================
// Same buffer interface as dequant_matvec_8bit (buffers 0-7).
// Buffer 2 (biases) accepted but IGNORED — MXFP8 has no biases.
//
// Weight layout: [out_dim, in_dim/4] uint32 (4 x FP8 E4M3 per uint32)
// Scale layout:  [out_dim, in_dim/group_size] uint8 stored as uint16 (padded)
//   NOTE: scales are E8M0 (uint8) but packed in the model_weights.bin as uint16 pairs
//         We read them as uint16 but only use the low byte

constant uint ROWS_PER_TG_MX = 4;

kernel void dequant_matvec_mxfp8(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint8_t*  scales     [[buffer(1)]],  // E8M0 scales (uint8, NOT uint16!)
    device const uint16_t* biases     [[buffer(2)]],  // UNUSED
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_MX + simd_group;
    uint packed_cols = in_dim / 4;
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint8_t*  s_row = scales + row * num_groups;
    (void)biases;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 4);
        float scale = e8m0_to_f32(s_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 4;

        float x0 = x_shared[x_base + 0];
        float x1 = x_shared[x_base + 1];
        float x2 = x_shared[x_base + 2];
        float x3 = x_shared[x_base + 3];

        // E4M3 dequant: convert byte to float, multiply by E8M0 scale
        acc += e4m3_to_f32((packed >>  0) & 0xFF) * scale * x0;
        acc += e4m3_to_f32((packed >>  8) & 0xFF) * scale * x1;
        acc += e4m3_to_f32((packed >> 16) & 0xFF) * scale * x2;
        acc += e4m3_to_f32((packed >> 24) & 0xFF) * scale * x3;
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}
