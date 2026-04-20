#!/usr/bin/env python3
"""Convert DFlash draft model (safetensors) to flat binary for C inference.
Output: draft_weights.bin — all weights concatenated as FP32.
         draft_weights.json — metadata (names, shapes, offsets).
"""
import json, struct, os, sys
import numpy as np

SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash"
    "/snapshots/c69b1858be3f89aa3c4714ddef516c976b9cc82d"
)
OUT_DIR = os.path.expanduser("~/flash-moe/metal_infer")

def main():
    # Load safetensors
    try:
        from safetensors import safe_open
    except ImportError:
        os.system("pip install safetensors -q")
        from safetensors import safe_open

    st_path = os.path.join(SNAPSHOT, "model.safetensors")
    if not os.path.exists(st_path):
        # Follow symlink
        st_path = os.path.realpath(st_path)
    print(f"Loading: {st_path}")

    # Collect all tensors sorted by name
    tensors = {}
    with safe_open(st_path, framework="pt") as f:
        import torch
        for key in sorted(f.keys()):
            t = f.get_tensor(key)
            # Convert bfloat16 → float32 via numpy
            if t.dtype == torch.bfloat16:
                tensors[key] = t.float().numpy()
            else:
                tensors[key] = t.numpy()

    # Print summary
    total_params = 0
    for name, arr in sorted(tensors.items()):
        total_params += arr.size
        print(f"  {name}: {arr.shape} ({arr.dtype}) = {arr.size:,} params")

    print(f"\nTotal: {total_params:,} params = {total_params*4/1e6:.1f} MB (FP32)")

    # Write flat binary + metadata
    bin_path = os.path.join(OUT_DIR, "draft_weights.bin")
    meta_path = os.path.join(OUT_DIR, "draft_weights_meta.json")

    metadata = {}
    offset = 0

    with open(bin_path, "wb") as bf:
        for name in sorted(tensors.keys()):
            arr = tensors[name].astype(np.float32)  # convert bf16 → fp32
            nbytes = arr.size * 4
            bf.write(arr.tobytes())
            metadata[name] = {
                "shape": list(arr.shape),
                "offset": offset,
                "size": arr.size,
                "nbytes": nbytes,
            }
            offset += nbytes
            print(f"  Written {name}: offset={metadata[name]['offset']}, {nbytes} bytes")

    with open(meta_path, "w") as mf:
        json.dump(metadata, mf, indent=2)

    total_mb = offset / 1e6
    print(f"\nDone! {bin_path}: {total_mb:.1f} MB")
    print(f"Metadata: {meta_path}")

if __name__ == "__main__":
    main()
