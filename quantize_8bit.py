#!/usr/bin/env python3
"""
quantize_8bit.py — Quantize bf16 model to 8-bit affine, one tensor at a time.

Unlike mlx_vlm.convert which loads the entire model into memory,
this script streams tensors individually. Works on machines with less RAM than model size.

Input:  Qwen/Qwen3.5-35B-A3B (bf16 safetensors, ~67 GB)
Output: Safetensors files with 8-bit quantized weights (U32 packed) + scales/biases (BF16)

Usage:
    python quantize_8bit.py --input ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/HASH --output ./qwen35-custom-8bit
"""

import argparse
import json
import struct
import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil

GROUP_SIZE = 64
BITS = 8

def bf16_to_f32(bf16_u16):
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(f32):
    return (f32.view(np.uint32) >> 16).astype(np.uint16)

def quantize_tensor_8bit(weight_bf16_u16, group_size=64):
    """Quantize a bf16 weight tensor to 8-bit affine.

    Input: [..., in_dim] as uint16 (bf16 bit pattern) — any number of leading dims
    Output: (packed_u32, scales_bf16, biases_bf16) with same leading dims
    """
    # Convert to float32
    weight_f32 = bf16_to_f32(weight_bf16_u16)
    original_shape = weight_f32.shape
    # Flatten all leading dims into one
    in_dim = original_shape[-1]
    out_dim = int(np.prod(original_shape[:-1]))
    weight_f32 = weight_f32.reshape(out_dim, in_dim)

    num_groups = in_dim // group_size
    vals_per_u32 = 32 // BITS  # 4 for 8-bit
    packed_cols = in_dim // vals_per_u32

    # Quantize per group
    scales = np.zeros((out_dim, num_groups), dtype=np.float32)
    biases = np.zeros((out_dim, num_groups), dtype=np.float32)
    packed = np.zeros((out_dim, packed_cols), dtype=np.uint32)

    max_val = (1 << BITS) - 1  # 255 for 8-bit

    for g in range(num_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        group_data = weight_f32[:, col_start:col_end]  # [out_dim, group_size]

        # Compute per-row scale and bias for this group
        g_min = group_data.min(axis=1)  # [out_dim]
        g_max = group_data.max(axis=1)  # [out_dim]

        s = (g_max - g_min) / max_val
        b = g_min

        # Handle zero range
        zero_mask = s == 0
        s[zero_mask] = 1.0

        scales[:, g] = s
        biases[:, g] = b

        # Quantize: uint_val = clamp(round((val - bias) / scale), 0, max_val)
        quant = np.clip(np.round((group_data - b[:, None]) / s[:, None]), 0, max_val).astype(np.uint8)

        # Pack into uint32 (4 values per uint32 for 8-bit)
        pack_start = g * (group_size // vals_per_u32)
        for p in range(group_size // vals_per_u32):
            val = np.zeros(out_dim, dtype=np.uint32)
            for n in range(vals_per_u32):
                val |= quant[:, p * vals_per_u32 + n].astype(np.uint32) << (n * BITS)
            packed[:, pack_start + p] = val

    # Restore original leading dims
    packed_shape = list(original_shape[:-1]) + [packed_cols]
    scales_shape = list(original_shape[:-1]) + [num_groups]
    return packed.reshape(packed_shape), f32_to_bf16(scales.reshape(scales_shape)), f32_to_bf16(biases.reshape(scales_shape))


def parse_safetensors_header(filepath):
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def read_tensor(filepath, meta, data_start):
    """Read a single tensor from safetensors file."""
    offset = meta['data_offsets'][0]
    size = meta['data_offsets'][1] - offset
    dtype_map = {'BF16': np.uint16, 'F32': np.float32, 'U32': np.uint32, 'I32': np.int32}
    np_dtype = dtype_map.get(meta['dtype'], np.uint8)

    with open(filepath, 'rb') as f:
        f.seek(data_start + offset)
        data = np.frombuffer(f.read(size), dtype=np_dtype)

    if len(meta['shape']) > 0:
        data = data.reshape(meta['shape'])
    return data


def should_quantize(name, shape):
    """Determine if a tensor should be quantized."""
    # Don't quantize: norms, biases, small tensors, 1D tensors
    if len(shape) < 2:
        return False
    if 'norm' in name.lower():
        return False
    if 'bias' in name.lower() and 'dt_bias' not in name:
        return False
    if 'A_log' in name:
        return False
    if 'dt_bias' in name:
        return False
    # Only quantize weight tensors (2D, large)
    total_elements = 1
    for s in shape:
        total_elements *= s
    if total_elements < 256:  # too small
        return False
    if shape[-1] % GROUP_SIZE != 0:  # in_dim must be divisible by group_size
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Quantize bf16 model to 8-bit')
    parser.add_argument('--input', required=True, help='Path to bf16 model directory')
    parser.add_argument('--output', default='./qwen35-custom-8bit', help='Output directory')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index_path = input_dir / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found")
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Group tensors by source file
    by_file = defaultdict(list)
    for name, filename in weight_map.items():
        by_file[filename].append(name)

    # Process each source file → write shards incrementally (low memory)
    SHARD_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB per shard
    shard_idx = 0
    current_shard = {}
    current_size = 0
    shard_files = []
    output_weight_map = {}
    total_original = 0
    total_quantized = 0

    def flush_shard():
        nonlocal shard_idx, current_shard, current_size
        if not current_shard:
            return
        shard_idx += 1
        shard_name = f"model-{shard_idx:05d}.safetensors"
        write_safetensors(output_dir / shard_name, current_shard)
        shard_files.append(shard_name)
        current_shard = {}
        current_size = 0

    for src_filename in sorted(by_file.keys()):
        src_path = input_dir / src_filename
        header, data_start = parse_safetensors_header(str(src_path))

        print(f"\nProcessing {src_filename} ({len(by_file[src_filename])} tensors)...")

        for name in sorted(by_file[src_filename]):
            if name == '__metadata__' or name not in header:
                continue

            meta = header[name]
            shape = meta['shape']
            dtype = meta['dtype']
            size = meta['data_offsets'][1] - meta['data_offsets'][0]
            total_original += size

            if should_quantize(name, shape) and dtype == 'BF16':
                data = read_tensor(str(src_path), meta, data_start)
                packed, scales, biases = quantize_tensor_8bit(data, GROUP_SIZE)
                del data  # free immediately

                pb, sb, bb = packed.tobytes(), scales.tobytes(), biases.tobytes()
                q_size = len(pb) + len(sb) + len(bb)
                total_quantized += q_size

                # Check shard limit
                if current_size + q_size > SHARD_SIZE:
                    flush_shard()

                current_shard[name] = (pb, list(packed.shape), 'U32')
                current_shard[name.replace('.weight', '.scales')] = (sb, list(scales.shape), 'BF16')
                current_shard[name.replace('.weight', '.biases')] = (bb, list(biases.shape), 'BF16')
                current_size += q_size
                output_weight_map[name] = f"model-{shard_idx + 1:05d}.safetensors"
                output_weight_map[name.replace('.weight', '.scales')] = output_weight_map[name]
                output_weight_map[name.replace('.weight', '.biases')] = output_weight_map[name]
                del packed, scales, biases, pb, sb, bb

                ratio = q_size / size * 100
                print(f"  Q {name}: {shape} ({size/1e6:.1f}MB → {q_size/1e6:.1f}MB, {ratio:.0f}%)")
            else:
                data = read_tensor(str(src_path), meta, data_start)
                db = data.tobytes()
                total_quantized += len(db)

                if current_size + len(db) > SHARD_SIZE:
                    flush_shard()

                current_shard[name] = (db, shape, dtype)
                current_size += len(db)
                output_weight_map[name] = f"model-{shard_idx + 1:05d}.safetensors"
                del data, db
                print(f"  C {name}: {shape} {dtype} ({size/1e6:.1f}MB)")

    flush_shard()  # write remaining

    # Write index
    index_out = {
        "metadata": {"total_size": total_quantized},
        "weight_map": output_weight_map
    }
    with open(output_dir / 'model.safetensors.index.json', 'w') as f:
        json.dump(index_out, f, indent=2)

    # Copy config and tokenizer files
    for fname in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                  'generation_config.json', 'chat_template.jinja',
                  'preprocessor_config.json', 'processor_config.json',
                  'video_preprocessor_config.json', 'vocab.json']:
        src = input_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(output_dir / fname))

    # Update config with quantization info
    config_path = output_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        cfg['quantization'] = {'group_size': GROUP_SIZE, 'bits': BITS, 'mode': 'affine'}
        cfg['quantization_config'] = cfg['quantization']
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Original: {total_original/1e9:.2f} GB → Quantized: {total_quantized/1e9:.2f} GB")
    print(f"Compression: {total_quantized/total_original*100:.1f}%")
    print(f"Output: {output_dir}")
    print(f"Shards: {total_shards}")


def write_safetensors(filepath, tensors):
    """Write tensors in safetensors format."""
    dtype_to_st = {'U32': 'U32', 'BF16': 'BF16', 'F32': 'F32', 'I32': 'I32'}

    # Build header
    header = {}
    offset = 0
    tensor_data = []

    for name in sorted(tensors.keys()):
        data_bytes, shape, dtype = tensors[name]
        header[name] = {
            'dtype': dtype_to_st.get(dtype, dtype),
            'shape': shape,
            'data_offsets': [offset, offset + len(data_bytes)]
        }
        tensor_data.append(data_bytes)
        offset += len(data_bytes)

    header_json = json.dumps(header).encode('utf-8')
    header_len = len(header_json)

    with open(filepath, 'wb') as f:
        f.write(struct.pack('<Q', header_len))
        f.write(header_json)
        for data in tensor_data:
            f.write(data)

    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  Wrote {filepath.name} ({size_mb:.1f} MB, {len(tensors)} tensors)")


if __name__ == '__main__':
    main()
