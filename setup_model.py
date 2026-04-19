#!/usr/bin/env python3
"""
setup_model.py — Download, extract, and pack Qwen3.5-35B-A3B-4bit for flash-moe inference.

This is the autoresearch-style one-shot setup: run once, then ./infer works.

Steps:
  1. Download model from HuggingFace (if not cached)
  2. Extract non-expert weights → model_weights.bin/json
  3. Generate expert_index.json
  4. Pack experts into per-layer binaries → packed_experts/

Usage:
    python setup_model.py                          # full setup
    python setup_model.py --skip-download          # if model already cached
    python setup_model.py --model-id <hf_model>    # use different model
"""

import json
import struct
import os
import sys
import time
import argparse
import re
from pathlib import Path
from collections import defaultdict

# Model architecture constants (Qwen3.5-35B-A3B-4bit)
NUM_EXPERTS = 256
NUM_LAYERS = 40
MOE_INTERMEDIATE = 512
HIDDEN_DIM = 2048
GROUP_SIZE = 64

# Expert component layouts — auto-selected by --bits flag
COMPONENTS_4BIT = [
    {"name": "gate_proj.weight",  "offset": 0,         "size": 524288,  "shape": [512, 256]},
    {"name": "gate_proj.scales",  "offset": 524288,    "size": 32768,   "shape": [512, 32]},
    {"name": "gate_proj.biases",  "offset": 557056,    "size": 32768,   "shape": [512, 32]},
    {"name": "up_proj.weight",    "offset": 589824,    "size": 524288,  "shape": [512, 256]},
    {"name": "up_proj.scales",    "offset": 1114112,   "size": 32768,   "shape": [512, 32]},
    {"name": "up_proj.biases",    "offset": 1146880,   "size": 32768,   "shape": [512, 32]},
    {"name": "down_proj.weight",  "offset": 1179648,   "size": 524288,  "shape": [2048, 64]},
    {"name": "down_proj.scales",  "offset": 1703936,   "size": 32768,   "shape": [2048, 8]},
    {"name": "down_proj.biases",  "offset": 1736704,   "size": 32768,   "shape": [2048, 8]},
]

COMPONENTS_8BIT = [
    {"name": "gate_proj.weight",  "offset": 0,         "size": 1048576, "shape": [512, 512]},
    {"name": "gate_proj.scales",  "offset": 1048576,   "size": 32768,   "shape": [512, 32]},
    {"name": "gate_proj.biases",  "offset": 1081344,   "size": 32768,   "shape": [512, 32]},
    {"name": "up_proj.weight",    "offset": 1114112,   "size": 1048576, "shape": [512, 512]},
    {"name": "up_proj.scales",    "offset": 2162688,   "size": 32768,   "shape": [512, 32]},
    {"name": "up_proj.biases",    "offset": 2195456,   "size": 32768,   "shape": [512, 32]},
    {"name": "down_proj.weight",  "offset": 2228224,   "size": 1048576, "shape": [2048, 128]},
    {"name": "down_proj.scales",  "offset": 3276800,   "size": 32768,   "shape": [2048, 8]},
    {"name": "down_proj.biases",  "offset": 3309568,   "size": 32768,   "shape": [2048, 8]},
]

# MXFP8 E4M3 format: same total size as 8-bit but NO biases, larger scales (group_size=32)
COMPONENTS_MXFP8 = [
    {"name": "gate_proj.weight",  "offset": 0,         "size": 1048576, "shape": [512, 512]},
    {"name": "gate_proj.scales",  "offset": 1048576,   "size": 65536,   "shape": [512, 64]},
    {"name": "up_proj.weight",    "offset": 1114112,   "size": 1048576, "shape": [512, 512]},
    {"name": "up_proj.scales",    "offset": 2162688,   "size": 65536,   "shape": [512, 64]},
    {"name": "down_proj.weight",  "offset": 2228224,   "size": 1048576, "shape": [2048, 128]},
    {"name": "down_proj.scales",  "offset": 3276800,   "size": 65536,   "shape": [2048, 16]},
]

EXPERT_SIZE_4BIT = 1769472
EXPERT_SIZE_8BIT = 3342336
EXPERT_SIZE = EXPERT_SIZE_4BIT  # default, overridden by --bits
COMPONENTS = COMPONENTS_4BIT     # default
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE


def parse_safetensors_header(filepath):
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def step1_download(model_id):
    """Download model from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"Step 1: Download {model_id}")
    print(f"{'='*60}")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id, local_dir=None)
        print(f"Model cached at: {path}")
        return path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Trying manual cache lookup...")
        # Try to find in cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir_name = f"models--{model_id.replace('/', '--')}"
        model_cache = os.path.join(cache_dir, model_dir_name)
        if os.path.exists(model_cache):
            snapshots = os.path.join(model_cache, "snapshots")
            if os.path.exists(snapshots):
                versions = sorted(os.listdir(snapshots))
                if versions:
                    path = os.path.join(snapshots, versions[-1])
                    print(f"Found cached model at: {path}")
                    return path
        raise


def step2_extract_weights(model_path, output_dir):
    """Extract non-expert weights into model_weights.bin."""
    print(f"\n{'='*60}")
    print(f"Step 2: Extract non-expert weights")
    print(f"{'='*60}")

    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        # Single safetensors file (non-sharded)
        st_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        if not st_files:
            print("ERROR: No safetensors files found")
            sys.exit(1)
        # Create a pseudo-index
        weight_map = {}
        for f in st_files:
            header, _ = parse_safetensors_header(os.path.join(model_path, f))
            for name in header:
                if name != '__metadata__':
                    weight_map[name] = f
    else:
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx['weight_map']

    expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}
    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            continue
        if expert_pattern.search(name):
            continue
        tensors_to_extract[name] = filename

    print(f"Extracting {len(tensors_to_extract)} non-expert tensors...")

    # Group by file
    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = os.path.join(model_path, filename)
        header_cache[filename] = parse_safetensors_header(filepath)

    def sanitize_name(name):
        if name.startswith("language_model."):
            return name[len("language_model."):]
        return name

    all_tensors = sorted([(sanitize_name(n), n, tensors_to_extract[n]) for n in tensors_to_extract])

    bin_path = os.path.join(output_dir, 'model_weights.bin')
    manifest = {"model": model_path, "num_tensors": len(all_tensors), "tensors": {},
                "config": {
                    "hidden_size": HIDDEN_DIM, "num_hidden_layers": NUM_LAYERS,
                    "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
                    "vocab_size": 248320, "num_experts": NUM_EXPERTS,
                    "num_experts_per_tok": 8, "moe_intermediate_size": MOE_INTERMEDIATE,
                    "full_attention_interval": 4,
                }}

    layer_types = ["full_attention" if (i+1) % 4 == 0 else "linear_attention" for i in range(NUM_LAYERS)]
    manifest["config"]["layer_types"] = layer_types

    ALIGN = 64
    offset = 0
    total_bytes = 0

    with open(bin_path, 'wb') as out_f:
        for i, (san_name, orig_name, filename) in enumerate(all_tensors):
            header, data_start = header_cache[filename]
            if orig_name not in header:
                continue
            meta = header[orig_name]
            tensor_offsets = meta['data_offsets']
            byte_len = tensor_offsets[1] - tensor_offsets[0]

            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b'\x00' * pad)
                offset += pad

            with open(os.path.join(model_path, filename), 'rb') as sf:
                sf.seek(data_start + tensor_offsets[0])
                data = sf.read(byte_len)

            out_f.write(data)
            manifest["tensors"][san_name] = {
                "offset": offset, "size": byte_len,
                "shape": meta['shape'], "dtype": meta['dtype']
            }
            offset += byte_len
            total_bytes += byte_len

            if (i + 1) % 50 == 0 or i == len(all_tensors) - 1:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e6:.1f} MB")

    json_path = os.path.join(output_dir, 'model_weights.json')
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {bin_path} ({total_bytes / 1e9:.2f} GB)")
    return bin_path


def step3_generate_expert_index(model_path, output_dir):
    """Generate expert_index.json mapping expert weights to safetensors locations."""
    print(f"\n{'='*60}")
    print(f"Step 3: Generate expert index")
    print(f"{'='*60}")

    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx['weight_map']
    else:
        st_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        weight_map = {}
        for fn in st_files:
            header, _ = parse_safetensors_header(os.path.join(model_path, fn))
            for name in header:
                if name != '__metadata__':
                    weight_map[name] = fn

    expert_pattern = re.compile(
        r'(?:language_model\.)?model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$'
    )

    expert_reads = {}
    header_cache = {}

    for tensor_name, filename in weight_map.items():
        m = expert_pattern.match(tensor_name)
        if not m:
            continue

        layer_idx = m.group(1)
        proj = m.group(2)   # gate_proj, up_proj, down_proj
        comp = m.group(3)   # weight, scales, biases
        comp_name = f"{proj}.{comp}"

        if filename not in header_cache:
            header_cache[filename] = parse_safetensors_header(os.path.join(model_path, filename))

        header, data_start = header_cache[filename]
        meta = header[tensor_name]
        tensor_offsets = meta['data_offsets']
        total_size = tensor_offsets[1] - tensor_offsets[0]
        shape = meta['shape']

        # For all-experts-in-one tensor: shape[0] = num_experts * per_expert_rows
        # Expert stride = total_size / num_experts
        expert_size = total_size // NUM_EXPERTS
        expert_stride = expert_size  # contiguous experts

        abs_offset = data_start + tensor_offsets[0]

        if layer_idx not in expert_reads:
            expert_reads[layer_idx] = {}

        expert_reads[layer_idx][comp_name] = {
            "file": filename,
            "abs_offset": abs_offset,
            "expert_stride": expert_stride,
            "expert_size": expert_size,
        }

    index_out = {
        "expert_reads": expert_reads,
        "model_path": model_path,
    }

    out_path = os.path.join(output_dir, 'expert_index.json')
    with open(out_path, 'w') as f:
        json.dump(index_out, f, indent=2)

    print(f"Generated {out_path}")
    print(f"Layers indexed: {len(expert_reads)}")
    for layer_key in sorted(expert_reads.keys(), key=int)[:3]:
        comps = expert_reads[layer_key]
        print(f"  Layer {layer_key}: {len(comps)} components")
        for comp_name, info in comps.items():
            print(f"    {comp_name}: stride={info['expert_stride']}, size={info['expert_size']}")

    return out_path


def step4_pack_experts(expert_index_path, output_dir, mxfp8=False):
    """Pack expert weights into per-layer binary files."""
    print(f"\n{'='*60}")
    print(f"Step 4: Pack experts into per-layer binaries")
    print(f"{'='*60}")

    with open(expert_index_path) as f:
        idx = json.load(f)
    expert_reads = idx['expert_reads']
    model_path = idx['model_path']

    suffix = "_mxfp8" if mxfp8 else ("_8bit" if EXPERT_SIZE == EXPERT_SIZE_8BIT else "")
    packed_dir = os.path.join(output_dir, f"packed_experts{suffix}")
    os.makedirs(packed_dir, exist_ok=True)

    # Open all needed source files
    needed_files = set()
    for layer_info in expert_reads.values():
        for info in layer_info.values():
            needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source files")

    total_written = 0
    t_start = time.monotonic()

    for layer_idx in range(NUM_LAYERS):
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"  Layer {layer_idx}: NOT in index, skipping")
            continue

        layer_info = expert_reads[layer_key]
        out_path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.bin")

        t0 = time.monotonic()
        fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd_out, LAYER_SIZE)

        # Build read plan sorted by source file + offset
        read_plan = []
        for expert_idx in range(NUM_EXPERTS):
            for comp in COMPONENTS:
                info = layer_info[comp['name']]
                src_fd = fds[info['file']]
                src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
                dst_offset = expert_idx * EXPERT_SIZE + comp['offset']
                read_plan.append((src_fd, src_offset, dst_offset, comp['size']))

        read_plan.sort(key=lambda x: (x[0], x[1]))

        bytes_written = 0
        for src_fd, src_offset, dst_offset, size in read_plan:
            data = os.pread(src_fd, size, src_offset)
            if len(data) != size:
                raise IOError(f"Short read: expected {size}, got {len(data)}")
            os.pwrite(fd_out, data, dst_offset)
            bytes_written += size

        os.close(fd_out)
        elapsed = time.monotonic() - t0
        total_written += bytes_written

        throughput = bytes_written / elapsed / 1e6 if elapsed > 0 else 0
        print(f"  Layer {layer_idx:2d}: {bytes_written/1e6:.1f} MB in {elapsed:.1f}s ({throughput:.0f} MB/s)")

    for fd in fds.values():
        os.close(fd)

    total_elapsed = time.monotonic() - t_start
    print(f"\nPacked {total_written/1e9:.2f} GB in {total_elapsed:.1f}s")
    print(f"Output: {packed_dir}")

    # Write layout
    layout = {"expert_size": EXPERT_SIZE, "num_layers": NUM_LAYERS,
              "num_experts": NUM_EXPERTS, "components": COMPONENTS}
    with open(os.path.join(packed_dir, "layout.json"), 'w') as f:
        json.dump(layout, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Setup Qwen3.5-35B-A3B for flash-moe inference")
    parser.add_argument('--model-id', default='mlx-community/Qwen3.5-122B-A10B-4bit',
                        help='HuggingFace model ID')
    parser.add_argument('--model-path', default=None,
                        help='Local model path (skip download)')
    parser.add_argument('--output', default='metal_infer',
                        help='Output directory')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--step', type=int, default=None,
                        help='Run only a specific step (1-4)')
    parser.add_argument('--bits', type=int, default=4, choices=[4, 8],
                        help='Quantization bits (4 or 8)')
    parser.add_argument('--mxfp8', action='store_true',
                        help='Use MXFP8 E4M3 format (no biases, group_size=32)')
    args = parser.parse_args()

    # Select constants based on quantization
    global COMPONENTS, EXPERT_SIZE, LAYER_SIZE, GROUP_SIZE
    if args.mxfp8:
        COMPONENTS = COMPONENTS_MXFP8
        EXPERT_SIZE = EXPERT_SIZE_8BIT  # same size!
        GROUP_SIZE = 32
        print(f"Quantization: MXFP8 E4M3, group_size=32, no biases, expert size: {EXPERT_SIZE:,} bytes")
    elif args.bits == 8:
        COMPONENTS = COMPONENTS_8BIT
        EXPERT_SIZE = EXPERT_SIZE_8BIT
    else:
        COMPONENTS = COMPONENTS_4BIT
        EXPERT_SIZE = EXPERT_SIZE_4BIT
    LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE
    if not args.mxfp8:
        print(f"Quantization: {args.bits}-bit, expert size: {EXPERT_SIZE:,} bytes")

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download
    if args.model_path:
        model_path = args.model_path
    elif args.skip_download:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir = f"models--{args.model_id.replace('/', '--')}"
        snapshots = os.path.join(cache_dir, model_dir, "snapshots")
        versions = sorted(os.listdir(snapshots))
        model_path = os.path.join(snapshots, versions[-1])
        print(f"Using cached model: {model_path}")
    else:
        if args.step and args.step != 1:
            print("Skipping download (--step specified)")
            model_path = None
        else:
            model_path = step1_download(args.model_id)

    if args.step is None or args.step == 2:
        step2_extract_weights(model_path, output_dir)

    expert_index_path = os.path.join(output_dir, 'expert_index.json')
    if args.step is None or args.step == 3:
        expert_index_path = step3_generate_expert_index(model_path, output_dir)

    if args.step is None or args.step == 4:
        step4_pack_experts(expert_index_path, output_dir, mxfp8=args.mxfp8)

    print(f"\n{'='*60}")
    print("Setup complete! Run inference with:")
    print(f"  cd metal_infer && ./infer --prompt 'Hello' --tokens 100")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
