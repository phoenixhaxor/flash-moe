#!/usr/bin/env python3
"""
Verify 8-bit expert weight dequantization by comparing:
1. Packed binary (packed_experts_8bit/layer_00.bin) → dequant in Python
2. Original MLX model safetensors → dequant via MLX

If these match, the packing is correct and the bug is in the C/Metal inference engine.
If they don't match, the packing itself is wrong.
"""

import os, struct, json, numpy as np

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-8bit/snapshots/q36"
)
PACKED_PATH = os.path.expanduser("~/flash-moe/metal_infer/packed_experts_8bit/layer_00.bin")
INDEX_PATH = os.path.expanduser("~/flash-moe/metal_infer/expert_index.json")

HIDDEN_DIM = 2048
MOE_INTERMEDIATE = 512
NUM_EXPERTS = 256
GROUP_SIZE = 64

# 8-bit expert layout offsets (must match infer.m)
GATE_W_OFF = 0
GATE_S_OFF = 1048576
GATE_B_OFF = 1081344
UP_W_OFF   = 1114112
UP_S_OFF   = 2162688
UP_B_OFF   = 2195456
DOWN_W_OFF = 2228224
DOWN_S_OFF = 3276800
DOWN_B_OFF = 3309568
EXPERT_SIZE = 3342336

def bf16_to_f32(raw_bytes):
    """Convert bfloat16 bytes to float32 numpy array."""
    # bfloat16 is 2 bytes, same mantissa as float32 but different exponent bias
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    # Convert bf16 to float32: shift left by 16 bits into a uint32, then reinterpret as float32
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)

def dequant_8bit_packed(w_raw, scales_raw, biases_raw, out_dim, in_dim, group_size):
    """
    Dequantize 8-bit packed weights (same logic as cpu_dequant_matvec in infer.m).
    w_raw: raw bytes of uint32 packed weights (4 x 8-bit values per uint32)
    scales_raw: raw bytes of bf16 scales
    biases_raw: raw bytes of bf16 biases
    """
    W = np.frombuffer(w_raw, dtype=np.uint32)
    scales = bf16_to_f32(scales_raw)
    biases = bf16_to_f32(biases_raw)
    
    vals_per_u32 = 4
    bits = 8
    mask = 0xFF
    packed_per_group = group_size // vals_per_u32
    packed_cols = in_dim // vals_per_u32
    num_groups = in_dim // group_size
    
    # Output: float32 matrix [out_dim, in_dim]
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    
    for row in range(out_dim):
        w_row = W[row * packed_cols : (row + 1) * packed_cols]
        s_row = scales[row * num_groups : (row + 1) * num_groups]
        b_row = biases[row * num_groups : (row + 1) * num_groups]
        
        for g in range(num_groups):
            scale = s_row[g]
            bias = b_row[g]
            base_packed = g * packed_per_group
            base_x = g * group_size
            
            for p in range(packed_per_group):
                packed = int(w_row[base_packed + p])
                x_base = base_x + p * vals_per_u32
                
                for n in range(vals_per_u32):
                    val = (packed >> (n * bits)) & mask
                    result[row, x_base + n] = float(val) * scale + bias
    
    return result

def read_packed_expert(expert_idx, component):
    """Read a specific expert's component from the packed binary."""
    base = expert_idx * EXPERT_SIZE
    
    if component == "gate.weight":
        return (base + GATE_W_OFF, 512 * 512 * 4)  # [512, 512] uint32
    elif component == "gate.scales":
        return (base + GATE_S_OFF, 512 * 32 * 2)   # [512, 32] bf16
    elif component == "gate.biases":
        return (base + GATE_B_OFF, 512 * 32 * 2)
    elif component == "up.weight":
        return (base + UP_W_OFF, 512 * 512 * 4)
    elif component == "up.scales":
        return (base + UP_S_OFF, 512 * 32 * 2)
    elif component == "up.biases":
        return (base + UP_B_OFF, 512 * 32 * 2)
    elif component == "down.weight":
        return (base + DOWN_W_OFF, 2048 * 128 * 4)  # [2048, 128] uint32
    elif component == "down.scales":
        return (base + DOWN_S_OFF, 2048 * 8 * 2)   # [2048, 8] bf16
    elif component == "down.biases":
        return (base + DOWN_B_OFF, 2048 * 8 * 2)

def parse_safetensors_header(filepath):
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start

def main():
    print("=" * 60)
    print("8-bit Expert Weight Verification")
    print("=" * 60)
    
    # ---- Step 1: Read from packed binary ----
    print("\n[1] Reading packed expert 0 from layer_00.bin...")
    with open(PACKED_PATH, 'rb') as f:
        # Read gate_proj for expert 0
        offset, size = read_packed_expert(0, "gate.weight")
        f.seek(offset)
        gate_w_packed = f.read(size)
        
        offset, size = read_packed_expert(0, "gate.scales")
        f.seek(offset)
        gate_s_packed = f.read(size)
        
        offset, size = read_packed_expert(0, "gate.biases")
        f.seek(offset)
        gate_b_packed = f.read(size)
    
    # Dequantize using same logic as C code
    print("  Dequantizing gate_proj (512x2048)...")
    gate_dequant_packed = dequant_8bit_packed(
        gate_w_packed, gate_s_packed, gate_b_packed,
        out_dim=512, in_dim=2048, group_size=64
    )
    print(f"  gate_proj: shape={gate_dequant_packed.shape}, "
          f"mean={gate_dequant_packed.mean():.6f}, std={gate_dequant_packed.std():.6f}, "
          f"min={gate_dequant_packed.min():.6f}, max={gate_dequant_packed.max():.6f}")
    print(f"  First 5 values of row 0: {gate_dequant_packed[0, :5]}")
    
    # ---- Step 2: Read from original MLX safetensors ----
    print("\n[2] Reading expert 0 gate_proj from MLX safetensors...")
    
    with open(os.path.join(MODEL_PATH, 'model.safetensors.index.json')) as f:
        idx = json.load(f)
    weight_map = idx['weight_map']
    
    # Find the tensor for layer 0, expert 0, gate_proj
    tensor_name = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    if tensor_name not in weight_map:
        print(f"  ERROR: {tensor_name} not found in weight_map")
        print(f"  Searching for similar names...")
        for k in sorted(weight_map.keys()):
            if 'layer.0' in k and 'gate' in k:
                print(f"    Found: {k}")
        return
    
    st_file = weight_map[tensor_name]
    st_path = os.path.join(MODEL_PATH, st_file)
    print(f"  File: {st_file}")
    
    header, data_start = parse_safetensors_header(st_path)
    meta = header[tensor_name]
    offsets = meta['data_offsets']
    shape = meta['shape']
    dtype = meta.get('dtype', 'uint32')
    total_size = offsets[1] - offsets[0]
    
    print(f"  Shape: {shape}, dtype: {dtype}, size: {total_size} bytes")
    
    # Read the raw weight data
    with open(st_path, 'rb') as f:
        f.seek(data_start + offsets[0])
        weight_data = f.read(total_size)
    
    # Expert stride
    expert_stride = total_size // NUM_EXPERTS
    print(f"  Total size: {total_size}, expert stride: {expert_stride}")
    
    # Expert 0's data
    expert_w_data = weight_data[0:expert_stride]
    print(f"  Expert 0 weight bytes: {len(expert_w_data)}")
    
    # Read scales and biases
    for suffix, comp_name in [(".scales", "scales"), (".biases", "biases")]:
        tname = f"model.layers.0.mlp.switch_mlp.gate_proj.{comp_name}"
        if tname in weight_map:
            sfile = weight_map[tname]
            sheader, sdata_start = parse_safetensors_header(os.path.join(MODEL_PATH, sfile))
            smeta = sheader[tname]
            soffsets = smeta['data_offsets']
            sshape = smeta['shape']
            sdtype = smeta.get('dtype', '')
            stotal = soffsets[1] - soffsets[0]
            print(f"  {comp_name}: shape={sshape}, dtype={sdtype}, size={stotal}")
    
    # ---- Step 3: Compare weight data byte-by-byte ----
    print("\n[3] Comparing expert 0 gate_proj.weight raw bytes...")
    
    # The packed binary stores expert data at expert_stride boundaries
    # Expert 0 is at offset 0 in the source, copied to offset 0 in packed
    # The packed gate.weight is at offset 0, size = 1048576 bytes
    # The source expert stride for weight = ?
    
    packed_w_size = 512 * 512 * 4  # 1,048,576 bytes
    print(f"  Packed weight size: {packed_w_size}")
    print(f"  Source weight size: {len(expert_w_data)}")
    
    if len(expert_w_data) == packed_w_size:
        # Direct byte comparison
        match = expert_w_data == gate_w_packed
        print(f"  Byte-for-byte match: {match}")
        if not match:
            # Find first mismatch
            for i in range(min(len(expert_w_data), len(gate_w_packed))):
                if expert_w_data[i] != gate_w_packed[i]:
                    print(f"  First mismatch at byte {i}: "
                          f"source=0x{expert_w_data[i]:02x} vs packed=0x{gate_w_packed[i]:02x}")
                    # Show context
                    ctx = 32
                    start = max(0, i - ctx)
                    end = min(len(expert_w_data), i + ctx)
                    print(f"  Source [{start}:{end}]: {expert_w_data[start:end].hex()}")
                    print(f"  Packed [{start}:{end}]: {gate_w_packed[start:end].hex()}")
                    break
    else:
        print(f"  SIZE MISMATCH! Source has {len(expert_w_data)} bytes but packed expects {packed_w_size}")
        print(f"  This means the MLX model stores weights in a different format than expected!")
        
        # Analyze what the source actually looks like
        # Maybe it's stored as uint8 instead of packed uint32?
        # For gate_proj [512, 2048] with 8-bit: if uint8, that's 512*2048 = 1,048,576 bytes
        # If uint32 packed (4 per u32), that's 512*512 = 262,144 uint32 = 1,048,576 bytes
        # SAME SIZE! So we can't distinguish by size alone.
        
        # Check if shape gives us a clue
        print(f"\n  Source tensor shape: {shape}")
        print(f"  Source tensor dtype: {dtype}")
        
        # If shape is [512, 2048] and dtype is uint8, each row has 2048 bytes
        # If shape is [512, 512] and dtype is uint32, each row has 512 uint32 = 2048 bytes
        # Either way, the raw bytes should be the same!
        
        # Let's check by trying to dequant from source directly
        # (treating it as the same format as packed)
        if dtype == 'uint8':
            print("\n  DETECTED: MLX stores weights as uint8 (not uint32 packed)")
            print("  The C code reads them as uint32 packed (4 x 8-bit per uint32)")
            print("  On little-endian ARM, this should be equivalent...")
            
            # But wait: if all 256 experts are in one tensor with shape [256, 512, 2048]
            # then expert_stride = 512*2048 = 1,048,576
            # This matches packed_w_size!
            
            # Let's dequant from source as if uint32 packed
            src_dequant = dequant_8bit_packed(
                expert_w_data, gate_s_packed, gate_b_packed,
                out_dim=512, in_dim=2048, group_size=64
            )
            print(f"  Source dequant: mean={src_dequant.mean():.6f}, std={src_dequant.std():.6f}")
            print(f"  Packed dequant: mean={gate_dequant_packed.mean():.6f}, std={gate_dequant_packed.std():.6f}")
            
            diff = np.abs(src_dequant - gate_dequant_packed)
            print(f"  Max abs diff: {diff.max():.6e}")
            print(f"  Mean abs diff: {diff.mean():.6e}")

    # ---- Step 4: Verify the actual values make sense ----
    print("\n[4] Sanity check: dequantized values distribution")
    print(f"  gate_proj row 0 first 10: {gate_dequant_packed[0, :10]}")
    print(f"  gate_proj row 0 last 10:  {gate_dequant_packed[0, -10:]}")
    print(f"  gate_proj row 100 first 10: {gate_dequant_packed[100, :10]}")

if __name__ == '__main__':
    main()
