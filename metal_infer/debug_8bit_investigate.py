#!/usr/bin/env python3
"""
Quick investigation: is row 100 truly zero? And full matvec sanity check.
Also verify non-expert weights (attention, norms) from model_weights.bin
"""
import os, struct, json, numpy as np

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-8bit/snapshots/q36"
)
PACKED_PATH = os.path.expandpath("~/flash-moe/metal_infer/packed_experts_8bit/layer_00.bin") if False else None
WEIGHTS_PATH = os.path.expanduser("~/flash-moe/metal_infer/model_weights.bin")
WEIGHTS_JSON = os.path.expanduser("~/flash-moe/metal_infer/model_weights.json")

def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)

def parse_header(filepath):
    with open(filepath, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen))
        return header, 8 + hlen

def main():
    # ====== Check row 100 of gate_proj expert 0 ======
    print("=== Investigating row 100 zeros ===")
    with open(os.path.expanduser("~/flash-moe/metal_infer/packed_experts_8bit/layer_00.bin"), 'rb') as f:
        # gate.weight at offset 0, expert 0
        f.seek(0)
        gate_w_raw = f.read(1048576)  # [512, 512] uint32
        f.seek(1048576)
        gate_s_raw = f.read(32768)    # [512, 32] bf16
        f.seek(1081344)
        gate_b_raw = f.read(32768)    # [512, 32] bf16
    
    W = np.frombuffer(gate_w_raw, dtype=np.uint32).reshape(512, 512)
    scales = bf16_to_f32(gate_s_raw).reshape(512, 32)
    biases = bf16_to_f32(gate_b_raw).reshape(512, 32)
    
    # Check row 100 raw uint32 values
    row100_w = W[100]
    row100_s = scales[100]
    row100_b = biases[100]
    print(f"  Row 100 weights (first 10 uint32): {row100_w[:10]}")
    print(f"  Row 100 scales (all 32): {row100_s}")
    print(f"  Row 100 biases (all 32): {row100_b}")
    
    # Dequant row 100 manually
    for g in range(3):  # first 3 groups
        s = row100_s[g]
        b = row100_b[g]
        packed_vals = row100_w[g*16:(g+1)*16]
        for p in range(4):  # first 4 packed
            packed = int(packed_vals[p])
            for n in range(4):
                val = (packed >> (n*8)) & 0xFF
                dq = val * s + b
                pos = g*64 + p*4 + n
                if dq != 0:
                    print(f"  Row100[{pos}] = {val} * {s:.6f} + {b:.6f} = {dq:.6f}")
    
    # Check how many rows are truly all-zero
    W_u8_all = np.zeros((512, 2048), dtype=np.float32)
    for n in range(4):
        v = ((W >> (n * 8)) & 0xFF).astype(np.float32)
        W_u8_all[:, n::4] = v
    scales_exp = np.repeat(scales, 16, axis=1)
    biases_exp = np.repeat(biases, 16, axis=1)
    dequant = W_u8_all * scales_exp + biases_exp  # but wait, this isn't right
    # Actually we need to be more careful
    
    # Proper dequant
    dequant = np.zeros((512, 2048), dtype=np.float32)
    for row in range(512):
        for g in range(32):
            s = float(scales[row, g])
            b = float(biases[row, g])
            for p in range(16):
                packed = int(W[row, g*16 + p])
                for n in range(4):
                    val = (packed >> (n*8)) & 0xFF
                    dequant[row, g*64 + p*4 + n] = val * s + b
    
    zero_rows = np.where(np.all(np.abs(dequant) < 1e-10, axis=1))[0]
    print(f"\n  Zero rows in dequantized gate_proj expert 0: {len(zero_rows)}")
    if len(zero_rows) > 0:
        print(f"  Row indices: {zero_rows[:20]}...")
        # Check if raw weights are zero too
        for r in zero_rows[:3]:
            raw_unique = np.unique(W[r])
            print(f"  Row {r}: unique raw uint32 values = {raw_unique[:5]}...")
    
    # ====== Check non-expert weights ======
    print("\n=== Checking non-expert weights from model_weights.bin ===")
    with open(WEIGHTS_JSON) as f:
        manifest = json.load(f)
    
    with open(WEIGHTS_PATH, 'rb') as f:
        data = f.read()
    
    print(f"  model_weights.bin size: {len(data)} bytes")
    print(f"  Number of tensors: {manifest['num_tensors']}")
    
    # Check key tensors for layer 0
    for tname in sorted(manifest['tensors'].keys()):
        if 'layers.0' in tname and ('layernorm' in tname or 'embed' in tname.lower()):
            info = manifest['tensors'][tname]
            offset = info['offset']
            size = info['size']
            shape = info['shape']
            dtype = info.get('dtype', '')
            raw = data[offset:offset+size]
            
            if 'bf16' in dtype.lower() or (len(shape) == 1 and 'norm' in tname):
                vals = bf16_to_f32(raw)
                print(f"  {tname}: shape={shape}, dtype={dtype}")
                print(f"    mean={vals.mean():.6f}, std={vals.std():.6f}")
                print(f"    first 5: {vals[:5]}")
            elif 'weight' in tname and 'embed' in tname.lower():
                # Embedding weight - check first token
                pass
    
    # Check embedding specifically
    for tname in ['model.embed_tokens.weight', 'model.embed_tokens.scales', 'model.embed_tokens.biases']:
        if tname in manifest['tensors']:
            info = manifest['tensors'][tname]
            offset = info['offset']
            size = info['size']
            shape = info['shape']
            raw = data[offset:offset+size]
            
            if 'scales' in tname or 'biases' in tname:
                vals = bf16_to_f32(raw)
                print(f"\n  {tname}: shape={shape}, values[:5]={vals[:5]}")
            else:
                # Weight - check shape
                print(f"\n  {tname}: shape={shape}, size={size}")
                # For 8-bit embedding: [248320, 512] uint32 = 248320 * 512 * 4 = 508,559,360 bytes
                print(f"    Expected size: {248320 * 512 * 4}")
    
    # Check layer 0 attention weights
    print("\n=== Layer 0 attention weight stats ===")
    for tname in sorted(manifest['tensors'].keys()):
        if 'layers.0' in tname and ('self_attn' in tname or 'linear_attn' in tname):
            info = manifest['tensors'][tname]
            offset = info['offset']
            size = info['size']
            shape = info['shape']
            dtype = info.get('dtype', '')
            raw = data[offset:offset+size]
            
            if 'weight' in tname:
                W_attn = np.frombuffer(raw, dtype=np.uint32)
                print(f"  {tname}: shape={shape}, dtype={dtype}, bytes={size}")
                # Quick stats: just look at unique values per uint32
                print(f"    unique values (sample): {np.unique(W_attn[:100])[:10]}")
            elif 'scales' in tname or 'biases' in tname:
                vals = bf16_to_f32(raw)
                print(f"  {tname}: shape={shape}, mean={vals.mean():.6f}, std={vals.std():.6f}")

if __name__ == '__main__':
    main()
