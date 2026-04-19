/*
 * shaders_mxfp8.metal — Metal compute shader for MXFP8 (E4M3) quantized matvec
 *
 * MXFP8 format:
 *   - Each weight is 1 byte = FP8 E4M3 (1 sign + 4 exponent + 3 mantissa)
 *   - 4 values packed per uint32
 *   - Per-group scale in bfloat16 (NO bias needed — floating point quantization)
 *   - group_size = 32 for expert weights
 *
 * Dequantization:
 *   value_fp32 = e4m3_to_fp32(uint8_value) * group_scale
 *
 * vs Affine uint8 (current):
 *   value_fp32 = uint8_value * scale + bias   (2 ops)
 *   value_fp32 = fp8_value * scale             (1 op + lookup)
 *
 * Author: Phoenix (for Tuan Andre / Panglima Ekspres)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// FP8 E4M3 to FP32 conversion
// ============================================================================
// E4M3 format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
// Special cases: exponent=0b1111 with mantissa=0b111 = NaN (rest are normal)
// Range: ±[2^-6 * 1.0,  2^9 * 1.875] = ±[0.015625, 960]
// Note: no infinity in E4M3 (unlike E5M2)

static inline float e4m3_to_f32(uint8_t v) {
    // Extract components
    uint sign     = (v >> 7) & 1;
    uint exponent = (v >> 3) & 0xF;
    uint mantissa = v & 0x7;

    if (exponent == 0 && mantissa == 0) {
        return 0.0f; // zero
    }

    // E4M3 exponent bias = 7
    // FP32 exponent bias = 127
    // FP32 exp = e4m3_exp - 7 + 127 = e4m3_exp + 120
    int fp32_exp = (int)exponent - 7 + 127;

    if (exponent == 0) {
        // Subnormal E4M3: value = (-1)^sign * 2^-6 * (mantissa/8)
        fp32_exp = 127 - 7; // = 120
        // mantissa is already < 8, so no implicit 1
        float val = (float)mantissa / 8.0f;
        // Adjust for subnormal: multiply by 2^(-6) relative to exp=1 case
        // exp=0 → actual exponent = 1-7 = -6, but without implicit 1
        val *= 0.5f; // compensate for missing implicit bit
        uint32_t bits = (sign << 31) | ((uint32_t)fp32_exp << 23) | ((uint32_t)(val * (1 << 23)) & 0x7FFFFF);
        float result;
        memcpy(&result, &bits, 4);
        return result;
    }

    // Normal: value = (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
    // FP32: sign(1) | exponent(8) | mantissa(23)
    // mantissa in FP32 = E4M3 mantissa << 20 (3 bits → 23 bits)
    uint32_t bits = (sign << 31) | ((uint32_t)fp32_exp << 23) | (mantissa << 20);

    // Handle NaN: E4M3 NaN when exponent=15 and mantissa=7
    if (exponent == 15 && mantissa == 7) {
        bits = 0x7FC00000; // FP32 NaN
        if (sign) bits |= (1u << 31);
    }

    float result;
    memcpy(&result, &bits, 4);
    return result;
}

// BF16 to FP32 conversion (reuse from main shaders)
static inline float bf16_to_f32_mxfp(uint16_t h) {
    uint32_t bits = ((uint32_t)h) << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

// ============================================================================
// Kernel: MXFP8 dequant matvec
// ============================================================================
// Same buffer interface as dequant_matvec_8bit (buffers 0-7) so infer.m doesn't need changes.
// Buffer 2 (biases) is accepted but IGNORED — MXFP8 has no biases.
//
// Layout:
//   W_packed: [out_dim, in_dim/4] uint32 (4 x FP8 per uint32)
//   scales:   [out_dim, in_dim/group_size] bfloat16
//   x:        [in_dim] float32
//   out:      [out_dim] float32

constant uint ROWS_PER_TG_MX = 4;

kernel void dequant_matvec_mxfp8(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/4]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // UNUSED — MXFP8 has no biases
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_MX + simd_group;
    uint packed_cols = in_dim / 4;      // 4 x FP8 values per uint32
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    (void)biases;  // suppress unused warning

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // Which group does this packed column belong to?
        uint g = col / (group_size / 4);
        float scale = bf16_to_f32_mxfp(s_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 4;

        // Extract 4 FP8 values and dequantize
        float x0 = x_shared[x_base + 0];
        float x1 = x_shared[x_base + 1];
        float x2 = x_shared[x_base + 2];
        float x3 = x_shared[x_base + 3];

        // E4M3 dequant: just convert to float and multiply by scale (no bias!)
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
