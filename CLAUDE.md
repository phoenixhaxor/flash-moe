# Flash-MoE: Running a 397B Parameter Model on a Laptop

> **[Read the paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.

Pure C/Metal inference engine that runs **Qwen3.5-397B-A17B** (a 397 billion parameter Mixture-of-Experts model) on a MacBook Pro with 48GB RAM at **5.5+ tokens/second** with production-quality output.

The entire 209GB model streams from SSD through a custom Metal compute pipeline. No Python. No frameworks. Just C, Objective-C, and hand-tuned Metal shaders.

## Results

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 2-bit experts, K=4 | **5.55** | Excellent | Current best. 120GB on disk. |
| 4-bit experts, K=4 (warm) | 4.80 | Excellent | 209GB on disk. Page-cache dependent. |
| 4-bit experts, K=4 (cold) | 2.83 | Excellent | Steady-state with cold cache. |
| Peak single token | **7.05** | — | Warm cache, 2-bit. |

## Hardware

- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)
- **macOS**: 26.2 (Darwin 25.2.0)

## Architecture

The model has 60 transformer layers: 45 GatedDeltaNet (linear attention) + 15 standard full attention. Each layer has 512 experts, of which K=4 are activated per token (plus one shared expert). Hidden dimension is 4096.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (120GB total at 2-bit) are read from NVMe SSD on demand via parallel `pread()`. Only the K=4 active experts per layer are loaded (~3.9MB each). Inspired by Apple's "LLM in a Flash" paper.

2. **2-bit Expert Quantization** — Custom requantization from MLX's 4-bit affine format to 2-bit affine (16 values per uint32). 44% size reduction with RMSE ~0.001. Quality preserved across math, code, and reasoning tasks.

3. **Metal Compute Shaders** — Hand-written Metal kernels for:
   - 4-bit and 2-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, shared input cache)
   - Fused SwiGLU activation
   - RMS normalization (two-pass: sum-of-squares reduction + apply)
   - Batched GPU attention (Q@K^T, softmax, scores@V) for full attention layers
   - GPU RoPE (fused with Q deinterleave and K normalization)
   - MoE combine + residual + sigmoid gate (fused kernel)

4. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer. The combine + residual + norm are also on GPU, feeding directly into the next layer's attention projections.

5. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head × 128×128 state matrix update. 64% faster than scalar code.

6. **F_NOCACHE for Direct SSD Access** — Bypasses the OS page cache for expert files when using 2-bit mode. With 120GB >> 35GB available cache, page caching thrashes. Direct I/O avoids eviction overhead.

### Pipeline Per Layer (3.14ms average at 2-bit)

```
CMD3(prev) → CMD1: attention projections  [0.87ms GPU]
           → CPU: GatedDeltaNet / full attention  [0.27ms CPU+BLAS]
           → CMD2: o_proj + residual + norm + routing + shared expert  [0.45ms GPU]
           → CPU: softmax + topK routing  [0.003ms]
           → I/O: parallel pread K=4 experts  [1.49ms SSD]
           → CMD3: expert forward + combine + norm (DEFERRED)  [0.03ms encode]
```

## Quick Start

```bash
cd metal_infer
make
# 4-bit inference (needs packed_experts/ directory)
./infer --prompt "Explain quantum computing" --tokens 100

# 2-bit inference (44% faster, needs packed_experts_2bit/)
./infer --prompt "Explain quantum computing" --tokens 100 --2bit

# Interactive chat
./chat --2bit
```

## Project Structure

```
metal_infer/
  infer.m              # Complete inference engine (~5000 lines)
  shaders.metal        # Metal compute kernels (~1100 lines)
  main.m               # MoE-only benchmark
  Makefile             # Build system
  extract_weights.py   # Creates model_weights.bin from safetensors
  encode_prompt.py     # Text → token IDs via HuggingFace tokenizer
  repack_experts_2bit.py  # 4-bit → 2-bit expert requantization
  model_weights.bin    # Non-expert weights (5.5GB, mmap'd)
  model_weights.json   # Tensor manifest
  vocab.bin            # Vocabulary for token decoding

stream_infer.py        # Reference Python/MLX implementation
repack_experts.py      # 4-bit expert packing from safetensors
progress.py            # Results visualization
results.tsv            # Experiment log
```

## What We Tried (and What Worked)

| Approach | Result | Verdict |
|----------|--------|---------|
| 2-bit expert quantization | +95% speed, quality preserved | **KEEP** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **KEEP** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78→0.28ms | **KEEP** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **KEEP** |
| GPU fused attention (RoPE kernels) | +2% for full-attn layers | **KEEP** |
| Pre-allocated Metal LRU cache (500) | 35% hit rate, marginal for 2-bit | Neutral |
| mmap expert files | 5x SLOWER (page fault overhead) | Reverted |
| Metal cache >500 entries | GPU memory pressure kills perf | Reverted |
| Malloc zero-copy cache (17GB) | Slower than Metal LRU | Reverted |
| Speculative early routing | Cache pollution + overhead | Reverted |
| GPU delta-net (195MB state) | Memory pressure > compute savings | Disabled |
| CMD1+CMD2 merge via GPU RoPE | Dispatch overhead > sync savings | Reverted |

## Safety

This is a primary development machine. The engine explicitly controls memory:
- Non-expert weights: 5.5GB (mmap'd, read-only)
- Metal scratch buffers: ~200MB
- Expert cache (optional): 0-3.5GB
- Total: 6-9GB, leaving 39-42GB for OS + page cache
- No OOM risk. Expert data streams from SSD on demand.
