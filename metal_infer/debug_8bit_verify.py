#!/usr/bin/env python3
"""
Verify 8-bit expert weights: compare packed binary vs original safetensors.
Focus on expert 0, gate_proj of layer 0.
"""
import os, struct, json, numpy as np

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-8bit/snapshots/q36"
)
PACKED_PATH = os.path.expanduser("~/flash-moe/metal_infer/packed_experts_8bit/layer_00.bin")

EXPERT_SIZE = 3342336
NUM_EXPERTS = 256

# 8-bit expert layout offsets (from infer.m)
GATE_W_OFF = 0
GATE_S_OFF = 1048576
GATE_B_OFF = 1081344
UP_W_OFF   = 1114112
UP_S_OFF   = 2162688
UP_B_OFF   = 2195456
DOWN_W_OFF = 2228224
DOWN_S_OFF = 3276800
DOWN_B_OFF = 3309568

def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)

def dequant_8bit(w_raw, s_raw, b_raw, out_dim, in_dim, group_size=64):
    W = np.frombuffer(w_raw, dtype=np.uint32)
    scales = bf16_to_f32(s_raw)
    biases = bf16_to_f32(b_raw)
    
    vals_per_u32 = 4
    bits = 8
    mask = 0xFF
    packed_per_group = group_size // vals_per_u32
    packed_cols = in_dim // vals_per_u32
    num_groups = in_dim // group_size
    
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for row in range(out_dim):
        w_off = row * packed_cols
        s_off = row * num_groups
        b_off = row * num_groups
        
        for g in range(num_groups):
            scale = float(scales[s_off + g])
            bias = float(biases[b_off + g])
            for p in range(packed_per_group):
                packed = int(W[w_off + g * packed_per_group + p])
                base = g * group_size + p * vals_per_u32
                for n in range(vals_per_u32):
                    val = (packed >> (n * bits)) & mask
                    result[row, base + n] = float(val) * scale + bias
    return result

def dequant_fast(w_raw, s_raw, b_raw, out_dim, in_dim, group_size=64):
    """Vectorized version for speed."""
    W = np.frombuffer(w_raw, dtype=np.uint32).reshape(out_dim, in_dim // 4)
    scales = bf16_to_f32(s_raw).reshape(out_dim, in_dim // group_size)
    biases = bf16_to_f32(b_raw).reshape(out_dim, in_dim // group_size)
    
    # Expand scale/bias to match packed columns
    # Each group has group_size/4 packed columns
    ppg = group_size // 4
    ng = in_dim // group_size
    scales_exp = np.repeat(scales, ppg, axis=1)  # [out_dim, in_dim/4]
    biases_exp = np.repeat(biases, ppg, axis=1)
    
    # Extract 4 values per uint32
    vals = np.zeros((out_dim, in_dim), dtype=np.float32)
    for n in range(4):
        v = ((W >> (n * 8)) & 0xFF).astype(np.float32)
        vals[:, n::4] = v * scales_exp + biases_exp
    
    return vals

def parse_header(filepath):
    with open(filepath, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen))
        return header, 8 + hlen

def main():
    print("=" * 60)
    print("8-bit Expert Weight Verification")
    print("=" * 60)
    
    # Load expert_index.json to get source locations
    idx_path = os.path.expanduser("~/flash-moe/metal_infer/expert_index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    
    layer_info = idx['expert_reads']['0']
    model_path = idx['model_path']
    
    # Weight map
    with open(os.path.join(model_path, 'model.safetensors.index.json')) as f:
        wm = json.load(f)['weight_map']
    
    # ====== Compare gate_proj for expert 0 ======
    print("\n--- Comparing gate_proj expert 0 ---")
    
    # From packed binary
    with open(PACKED_PATH, 'rb') as f:
        f.seek(0 * EXPERT_SIZE + GATE_W_OFF)
        packed_w = f.read(1048576)
        f.seek(0 * EXPERT_SIZE + GATE_S_OFF)
        packed_s = f.read(32768)
        f.seek(0 * EXPERT_SIZE + GATE_B_OFF)
        packed_b = f.read(32768)
    
    # From safetensors
    tname = 'language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight'
    st_file = wm[tname]
    header, data_start = parse_header(os.path.join(model_path, st_file))
    meta = header[tname]
    offsets = meta['data_offsets']
    total = offsets[1] - offsets[0]
    stride = total // 256
    
    with open(os.path.join(model_path, st_file), 'rb') as f:
        f.seek(data_start + offsets[0])
        all_w = f.read(total)
    
    src_w = all_w[0:stride]  # expert 0
    
    # Compare raw bytes
    print(f"  Source bytes: {len(src_w)}, Packed bytes: {len(packed_w)}")
    byte_match = src_w == packed_w
    print(f"  Byte-for-byte match: {byte_match}")
    
    if not byte_match:
        for i in range(len(src_w)):
            if src_w[i] != packed_w[i]:
                print(f"  FIRST MISMATCH at byte {i}: src=0x{src_w[i]:02x} packed=0x{packed_w[i]:02x}")
                ctx = 16
                s = max(0, i-ctx)
                e = min(len(src_w), i+ctx)
                print(f"  Source[{s}:{e}]: {src_w[s:e].hex()}")
                print(f"  Packed[{s}:{e}]: {packed_w[s:e].hex()}")
                break
    
    # Now dequant from source and compare values
    print("\n--- Dequantizing from source (safetensors) ---")
    src_gate = dequant_fast(src_w, packed_s, packed_b, 512, 2048)
    print(f"  Shape: {src_gate.shape}")
    print(f"  Mean: {src_gate.mean():.6f}, Std: {src_gate.std():.6f}")
    print(f"  Min: {src_gate.min():.6f}, Max: {src_gate.max():.6f}")
    print(f"  Row 0[:8]: {src_gate[0,:8]}")
    print(f"  Row 100[:8]: {src_gate[100,:8]}")
    
    # ====== Verify with MLX model directly ======
    print("\n--- Loading full model via MLX for reference ---")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    
    mx_weights = mx.load(os.path.join(model_path, st_file), stream=mx.cpu)
    gate_w_mx = mx_weights[tname]
    gate_w_np = np.array(gate_w_mx)
    
    print(f"  MLX gate_proj.weight shape: {gate_w_np.shape}, dtype: {gate_w_np.dtype}")
    print(f"  Expert 0 shape: {gate_w_np[0].shape}")
    
    # The MLX weights are U32 packed. Expert 0 is gate_w_np[0]
    expert0_w = gate_w_np[0]  # shape [512, 512] uint32
    expert0_w_bytes = expert0_w.tobytes()
    
    # Compare bytes
    mlx_vs_packed = expert0_w_bytes == packed_w
    mlx_vs_source = expert0_w_bytes == src_w
    print(f"  MLX expert0 vs packed: {mlx_vs_packed}")
    print(f"  MLX expert0 vs source: {mlx_vs_source}")
    
    # ====== Full dequant verification ======
    print("\n--- Full dequant with MLX native ---")
    # Load scales and biases
    for comp in ['scales', 'biases']:
        tname_s = f'language_model.model.layers.0.mlp.switch_mlp.gate_proj.{comp}'
        sfile = wm[tname_s]
        mx_s = mx.load(os.path.join(model_path, sfile), stream=mx.cpu)
        s_data = np.array(mx_s[tname_s])
        print(f"  {comp}: shape={s_data.shape}, dtype={s_data.dtype}")
        print(f"    Expert 0: {comp} shape={s_data[0].shape}")
        
        if comp == 'scales':
            mlx_scales = s_data[0]  # [512, 32]
        else:
            mlx_biases = s_data[0]  # [512, 32]
    
    # Dequant using MLX approach (should match what the model actually computes)
    # MLX uses: dequantized = weight * scale + bias (per group)
    # For 8-bit U32 packed: extract 4 values per uint32
    
    expert0_w_u32 = expert0_w  # [512, 512] uint32
    dequant_mlx = np.zeros((512, 2048), dtype=np.float32)
    
    for row in range(512):
        for g in range(32):  # 32 groups of 64
            scale = float(mlx_scales[row, g])
            bias = float(mlx_biases[row, g])
            for p in range(16):  # 16 packed per group (64/4)
                packed = int(expert0_w_u32[row, g*16 + p])
                base = g * 64 + p * 4
                for n in range(4):
                    val = (packed >> (n * 8)) & 0xFF
                    dequant_mlx[row, base + n] = float(val) * scale + bias
    
    print(f"\n  MLX dequant: mean={dequant_mlx.mean():.6f}, std={dequant_mlx.std():.6f}")
    print(f"  MLX dequant: min={dequant_mlx.min():.6f}, max={dequant_mlx.max():.6f}")
    print(f"  Row 0[:8]: {dequant_mlx[0,:8]}")
    
    # Compare our dequant vs MLX dequant
    diff = np.abs(dequant_mlx - src_gate)
    print(f"\n  Max diff (MLX dequant vs our dequant): {diff.max():.2e}")
    print(f"  Mean diff: {diff.mean():.2e}")
    
    # ====== Now do a matvec test ======
    print("\n--- Matvec test with random input ---")
    np.random.seed(42)
    x = np.random.randn(2048).astype(np.float32) * 0.01  # simulate hidden state
    
    # Using dequantized matrix
    result_ours = src_gate @ x
    
    print(f"  Input: mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"  Matvec result: shape={result_ours.shape}")
    print(f"  Result: mean={result_ours.mean():.6f}, std={result_ours.std():.6f}")
    print(f"  Result[:8]: {result_ours[:8]}")
    
    # ====== Check if expert_index.json strides match ======
    print("\n--- Verify expert_index.json layout ---")
    for comp_name in ['gate_proj.weight', 'gate_proj.scales', 'gate_proj.biases']:
        info = layer_info[comp_name]
        print(f"  {comp_name}: file={info['file']}, abs_offset={info['abs_offset']}, "
              f"stride={info['expert_stride']}, size={info['expert_size']}")

if __name__ == '__main__':
    main()
