#!/usr/bin/env python3
"""
Compute embedding for token_id=9419 ("Hello") from the MLX safetensors file.
Compare with the C engine's debug output.

C engine output:
  [DEBUG] token_id=9419
  [DEBUG] EMBEDDING: mean=0.00001741 std=0.01347606 min=-0.04541016 max=0.04589844
  [DEBUG] EMBEDDING first20: 0.01309586 0.00294876 0.00675392 0.01613998 -0.00339317 ...
"""
import struct, os, json, numpy as np

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-8bit/snapshots/q36"
)

def bf16_to_f32(u16_arr):
    u32 = u16_arr.astype(np.uint32) << 16
    return u32.view(np.float32)

def parse_header(filepath):
    with open(filepath, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen))
        return header, 8 + hlen

# Load weight map
with open(os.path.join(MODEL_PATH, 'model.safetensors.index.json')) as f:
    wm = json.load(f)['weight_map']

TOKEN_ID = 9419
HIDDEN_DIM = 2048
GROUP_SIZE = 64

# Find embed_tokens tensors
for suffix in ['weight', 'scales', 'biases']:
    tname = f'language_model.model.embed_tokens.{suffix}'
    if tname not in wm:
        print(f"ERROR: {tname} not found")
        exit(1)

# Read weight row for token_id
tname_w = 'language_model.model.embed_tokens.weight'
fn = wm[tname_w]
header, data_start = parse_header(os.path.join(MODEL_PATH, fn))
meta = header[tname_w]
shape = meta['shape']
dtype = meta['dtype']
total = meta['data_offsets'][1] - meta['data_offsets'][0]
offsets = meta['data_offsets']

print(f"embed_tokens.weight: shape={shape}, dtype={dtype}")
print(f"  bytes={total}, per_row={total // shape[0]}")

# Each row: shape[0]=248320 tokens, shape[1]=512 uint32 = 2048 packed 8-bit values
row_size = total // shape[0]  # 512 * 4 = 2048 bytes
print(f"  row_size={row_size} bytes")

with open(os.path.join(MODEL_PATH, fn), 'rb') as f:
    f.seek(data_start + offsets[0] + TOKEN_ID * row_size)
    w_row = f.read(row_size)

W = np.frombuffer(w_row, dtype=np.uint32)  # 512 uint32
print(f"  W shape: {W.shape}, first 5: {W[:5]}")

# Read scales
tname_s = 'language_model.model.embed_tokens.scales'
fn_s = wm[tname_s]
header_s, ds_s = parse_header(os.path.join(MODEL_PATH, fn_s))
meta_s = header_s[tname_s]
total_s = meta_s['data_offsets'][1] - meta_s['data_offsets'][0]
row_size_s = total_s // meta_s['shape'][0]
print(f"embed_tokens.scales: shape={meta_s['shape']}, row_size={row_size_s}")

with open(os.path.join(MODEL_PATH, fn_s), 'rb') as f:
    f.seek(ds_s + meta_s['data_offsets'][0] + TOKEN_ID * row_size_s)
    s_row = f.read(row_size_s)

scales = bf16_to_f32(np.frombuffer(s_row, dtype=np.uint16))  # 32 bf16
print(f"  scales: {scales[:5]}")

# Read biases
tname_b = 'language_model.model.embed_tokens.biases'
fn_b = wm[tname_b]
header_b, ds_b = parse_header(os.path.join(MODEL_PATH, fn_b))
meta_b = header_b[tname_b]
total_b = meta_b['data_offsets'][1] - meta_b['data_offsets'][0]
row_size_b = total_b // meta_b['shape'][0]

with open(os.path.join(MODEL_PATH, fn_b), 'rb') as f:
    f.seek(ds_b + meta_b['data_offsets'][0] + TOKEN_ID * row_size_b)
    b_row = f.read(row_size_b)

biases = bf16_to_f32(np.frombuffer(b_row, dtype=np.uint16))  # 32 bf16
print(f"  biases: {biases[:5]}")

# Dequantize: same logic as C code (8-bit: 4 vals per uint32)
# group_size=64, num_groups=32, packed_per_group=64/4=16, packed_cols=512
num_groups = 32
packed_per_group = 16
vals_per_u32 = 4
bits = 8
mask = 0xFF

out = np.zeros(HIDDEN_DIM, dtype=np.float32)
for g in range(num_groups):
    scale = float(scales[g])
    bias = float(biases[g])
    for p in range(packed_per_group):
        packed = int(W[g * packed_per_group + p])
        base = g * GROUP_SIZE + p * vals_per_u32
        for n in range(vals_per_u32):
            val = (packed >> (n * bits)) & mask
            out[base + n] = float(val) * scale + bias

print(f"\n=== Python Embedding for token_id={TOKEN_ID} ===")
print(f"  mean={out.mean():.8f} std={out.std():.8f} min={out.min():.8f} max={out.max():.8f}")
print(f"  first20: {' '.join(f'{v:.8f}' for v in out[:20])}")

# Compare with C output
c_first20 = [0.01309586, 0.00294876, 0.00675392, 0.01613998, -0.00339317,
             -0.01785278, -0.02774620, 0.00802231, -0.01785278, 0.01157379,
             0.00370979, 0.00345612, -0.02901459, 0.00218773, -0.00694466,
             -0.01759911, 0.00650024, 0.00066566, -0.00187111, -0.00770569]

print(f"\n=== Comparison with C engine ===")
print(f"  Python mean={out.mean():.8f}  C mean=0.00001741")
print(f"  Python std ={out.std():.8f}  C std =0.01347606")
print(f"  Python min ={out.min():.8f}  C min =-0.04541016")
print(f"  Python max ={out.max():.8f}  C max =0.04589844")

print(f"\n  first20 diff:")
all_match = True
for i in range(20):
    diff = abs(out[i] - c_first20[i])
    match = "✓" if diff < 1e-5 else "✗"
    if diff >= 1e-5:
        all_match = False
    print(f"    [{i:2d}] Python={out[i]:.8f}  C={c_first20[i]:.8f}  diff={diff:.2e} {match}")

if all_match:
    print("\n  ✅ ALL VALUES MATCH — embedding is correct!")
else:
    print("\n  ❌ MISMATCH DETECTED — embedding dequant differs!")
