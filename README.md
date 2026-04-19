# flash-moe (Multi-Model Fork)

Pure C/Metal inference engine that streams Qwen3.5/Qwen3.6 Mixture-of-Experts models from SSD on Apple Silicon Macs. Models much larger than available RAM run by keeping only attention weights in memory and streaming expert weights from disk on demand.

> Fork of [danveloper/flash-moe](https://github.com/danveloper/flash-moe) adapted for multiple Qwen3.5/3.6 MoE model sizes. Inspired by [Karpathy's autoresearch](https://github.com/karpathy) and Apple's [\"LLM in a Flash\"](https://arxiv.org/abs/2312.11514) paper (Alizadeh et al.).

## What This Is

- Runs the **122B parameter** Qwen3.5 MoE model on a Mac with **24GB RAM** using SSD streaming
- Pure C/Objective-C + Metal compute shaders -- no Python frameworks needed at runtime
- Expert weights (the bulk of the model) stream from NVMe SSD per token via parallel `pread()`
- Non-expert weights (embedding, attention, norms) stay resident in RAM (~3-5 GB)
- Custom 2-bit expert quantization cuts disk footprint by ~44% and nearly doubles throughput
- OpenAI-compatible HTTP server mode for drop-in integration

## Supported Models

| Model | Total Params | Active Params | 4-bit Disk | 8-bit Disk | 2-bit Disk | Speed (M4 Pro 24GB) |
|-------|-------------|---------------|-----------|-----------|-----------|---------------------|
| Qwen3.5-35B-A3B | 35B | 3B | ~18 GB | — | ~10 GB | ~11 / 20 tok/s |
| Qwen3.6-35B-A3B | 35B | 3B | — | ~18 GB | — | ~17 / 20 tok/s |
| Qwen3.5-122B-A10B | 122B | 10B | ~65 GB | — | ~36 GB | ~3 / 6.5 tok/s |

Speed format: prefill / decode tokens per second (approximate, varies by prompt and cache state).

## Hardware Requirements

- **Apple Silicon Mac** (M1, M2, M3, M4 -- any variant)
- **Minimum 16 GB RAM** (24 GB+ recommended for the 122B model)
- **SSD with enough free space** for the packed expert files (see table above)
- **macOS 14+** (Sonoma or later)

The engine uses Metal compute shaders for GPU acceleration and Apple's Accelerate framework for BLAS operations. No third-party C/C++ dependencies.

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/andrevirgantara/flash-moe.git
cd flash-moe
```

### 2. Install Python dependencies (setup only)

```bash
pip install mlx-lm mlx-vlm huggingface-hub safetensors numpy
```

These are only needed during model preparation. The inference engine itself is pure C/Metal.

### 3. Download and prepare a model

```bash
python setup_model.py --model-id mlx-community/Qwen3.5-122B-A10B-4bit
```

This single command will:
1. Download the model from HuggingFace (cached in `~/.cache/huggingface/`)
2. Extract non-expert weights into `metal_infer/model_weights.bin` + `.json`
3. Generate `metal_infer/expert_index.json`
4. Pack expert weights into `metal_infer/packed_experts/` (one file per layer)

For the 35B model, use `--model-id mlx-community/Qwen3.5-35B-A3B-4bit` instead.

For **Qwen3.6 8-bit** (recommended — higher quality, same speed):

```bash
python setup_model.py --model-id mlx-community/Qwen3.6-35B-A3B-8bit --bits 8
```

You can also run individual steps if the download is already cached:

```bash
python setup_model.py --skip-download                    # skip HuggingFace download
python setup_model.py --model-path /path/to/local/model  # use a local checkout
python setup_model.py --step 4                           # re-run only expert packing
```

### 4. Build the inference engine

```bash
cd metal_infer
make infer
```

### 5. Generate vocab.bin

Generate the vocabulary lookup file from the model's tokenizer:

```bash
python -c "
import json, struct, os
model_path = os.path.expanduser('~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-122B-A10B-4bit/snapshots')
snap = sorted(os.listdir(model_path))[-1]
tok = json.load(open(os.path.join(model_path, snap, 'tokenizer.json')))
vocab = sorted(tok['model']['vocab'].items(), key=lambda x: x[1])
max_id = vocab[-1][1]
with open('vocab.bin', 'wb') as f:
    f.write(struct.pack('<II', len(vocab), max_id))
    for s, i in vocab:
        b = s.encode('utf-8')
        f.write(struct.pack('<H', len(b)) + b)
print(f'Wrote vocab.bin: {len(vocab)} tokens, max_id={max_id}')
"
```

### 6. Generate tokenizer.bin

```bash
python export_tokenizer.py /path/to/model/tokenizer.json tokenizer.bin
```

Replace `/path/to/model/tokenizer.json` with the actual path to the downloaded model's `tokenizer.json` file (typically under `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-122B-A10B-4bit/snapshots/<hash>/tokenizer.json`).

### 7. Symlink packed experts (if running from metal_infer/)

If `setup_model.py` placed `packed_experts/` inside `metal_infer/` already, skip this step. Otherwise, create a symlink so the engine can find them:

```bash
ln -s /path/to/packed_experts packed_experts
```

### 8. Run inference

**4-bit / 2-bit models (Qwen3.5):**

```bash
./infer --prompt "Hello, what is quantum computing?" --tokens 100 --k 8
```

**8-bit models (Qwen3.6):**

```bash
./infer --8bit --prompt "Hello, what is quantum computing?" --tokens 100
```

Key flags:

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Input prompt text |
| `--tokens N` | Maximum tokens to generate (default: 20) |
| `--k N` | Active experts per layer (default: 4, model default: 8) |
| `--2bit` | Use 2-bit quantized experts (faster, requires `packed_experts_2bit/`) |
| `--8bit` | Use 8-bit MLX quantized model (Qwen3.6, requires `packed_experts_8bit/`) |
| `--model PATH` | Model directory path (default: current directory) |
| `--serve PORT` | Start OpenAI-compatible HTTP server on PORT |
| `--timing` | Print per-layer timing breakdown |
| `--think-budget N` | Max thinking tokens before forcing `</think>` (default: 2048) |

## 2-bit Expert Optimization

The 2-bit requantization reduces expert file sizes by ~44% and nearly doubles throughput because SSD I/O is the bottleneck. Quality is preserved (RMSE ~0.001 vs 4-bit).

### Generate 2-bit experts

```bash
cd metal_infer
python repack_experts_2bit.py --verify
```

This reads from `packed_experts/` and writes to `packed_experts_2bit/`. The `--verify` flag checks reconstruction error for the first few experts per layer.

### Run with 2-bit experts

```bash
./infer --prompt "Hello" --tokens 100 --k 8 --2bit
```

The engine auto-detects 2-bit experts if `packed_experts_2bit/` exists and `packed_experts/` does not.

## Server Mode (OpenAI-compatible)

Start the server:

```bash
./serve.sh start
```

This launches the inference engine as an HTTP server on port 8080 with auto-restart on crash.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming SSE) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

### Usage

```bash
# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-122b", "messages": [{"role": "user", "content": "Hello"}]}'

# Or point any OpenAI-compatible client at:
#   API Base: http://localhost:8080
#   Model:    qwen3.5-122b
```

### Server management

```bash
./serve.sh stop      # stop the server
./serve.sh restart   # restart
./serve.sh status    # check if running
./serve.sh logs      # tail the log file
```

### Interactive chat

There is also a standalone chat TUI (connects to the server):

```bash
cd metal_infer
make chat
./chat --k 8
```

## Adapting for New Models

To add support for a different Qwen3.5 MoE variant, update the model constants in `metal_infer/infer.m`:

```c
// These defines at the top of infer.m control the model architecture:
#define HIDDEN_DIM          3072    // model.config.hidden_size
#define NUM_LAYERS          48      // model.config.num_hidden_layers
#define NUM_ATTN_HEADS      32      // model.config.num_attention_heads
#define NUM_KV_HEADS        2       // model.config.num_key_value_heads
#define HEAD_DIM            256     // model.config.head_dim
#define VOCAB_SIZE          248320  // model.config.vocab_size
#define NUM_EXPERTS         256     // model.config.num_experts
#define NUM_EXPERTS_PER_TOK 8       // model.config.num_experts_per_tok
#define MOE_INTERMEDIATE    1024    // model.config.moe_intermediate_size
#define SHARED_INTERMEDIATE 1024    // model.config.shared_expert_intermediate_size
#define FULL_ATTN_INTERVAL  4       // model.config.full_attention_interval
```

You will also need to update `EXPERT_SIZE` and the expert layout offset constants (`GATE_W_OFF_4`, etc.) to match the new model's tensor shapes. The `setup_model.py` script has corresponding constants (`NUM_EXPERTS`, `NUM_LAYERS`, `MOE_INTERMEDIATE`, `HIDDEN_DIM`, `GROUP_SIZE`) that must match.

The linear attention constants (`LINEAR_NUM_V_HEADS`, `LINEAR_KEY_DIM`, etc.) and full attention constants (`ROPE_THETA`, `PARTIAL_ROTARY`) may also differ between model sizes -- check the model's `config.json`.

## Architecture

### SSD Streaming Pipeline

The core idea: a 122B-parameter MoE model has 256 experts per layer, but only K (typically 4-8) are activated per token. The non-expert weights (embedding, attention projections, norms, routing gates, shared expert) fit in ~3-5 GB of RAM. The expert weights (the other ~60 GB at 4-bit, ~36 GB at 2-bit) stream from SSD on demand.

```
Per-layer pipeline (~3.14ms average at 2-bit, K=4):

CMD3(prev) --> CMD1: attention projections           [0.87ms GPU]
           --> CPU:  GatedDeltaNet / full attention   [0.27ms CPU+BLAS]
           --> CMD2: o_proj + residual + norm +        [0.45ms GPU]
                     routing + shared expert
           --> CPU:  softmax + top-K routing           [0.003ms]
           --> I/O:  parallel pread K=4 experts        [1.49ms SSD]
           --> CMD3: expert forward + combine + norm   [0.03ms encode, deferred]
```

### Key techniques

1. **SSD Expert Streaming** -- Expert weights are read from NVMe SSD on demand via parallel `pread()` calls (one pthread per expert). Only the K active experts per layer are loaded. The OS page cache provides natural caching of recently-used experts.

2. **2-bit Expert Quantization** -- Custom requantization from MLX's 4-bit affine format to 2-bit affine (16 values per uint32 instead of 8 per uint32). Group size of 64 is preserved. ~44% size reduction, quality preserved.

3. **Metal Compute Shaders** -- Hand-written Metal kernels for dequantized matrix-vector multiply (4-bit and 2-bit), fused SwiGLU, RMS normalization, batched GPU attention, GPU RoPE, and MoE combine + residual + sigmoid gate.

4. **Deferred GPU Expert Compute** -- CMD3 (expert forward pass) is submitted without waiting for completion. The GPU executes it while the CPU prepares the next layer. The combine, residual add, and norm are also on GPU, feeding directly into the next layer's attention projections.

5. **Accelerate BLAS for Linear Attention** -- The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the state matrix update, 64% faster than scalar code.

6. **F_NOCACHE for Direct SSD Access** -- Bypasses the OS page cache for expert files in 2-bit mode, where the working set exceeds available cache. Avoids page cache thrashing.

### Memory budget

| Component | Size |
|-----------|------|
| Non-expert weights (mmap'd) | ~3-5 GB |
| Metal scratch buffers | ~200 MB |
| Expert LRU cache (optional) | 0-3.5 GB |
| **Total engine footprint** | **~3-9 GB** |

Expert data streams from SSD on demand and does not accumulate in RAM. No OOM risk.

## Project Structure

```
flash-moe/
  setup_model.py             # One-shot model download + preparation
  repack_experts.py          # Pack experts from safetensors to per-layer binaries
  serve.sh                   # Server launcher with auto-restart

  metal_infer/
    infer.m                  # Complete inference engine (~6500 lines Obj-C)
    shaders.metal            # Metal compute kernels
    chat.m                   # Interactive chat TUI (HTTP/SSE client)
    main.m                   # MoE-only benchmark harness
    Makefile                 # Build system
    extract_weights.py       # Create model_weights.bin from safetensors
    export_tokenizer.py      # Convert tokenizer.json to tokenizer.bin
    repack_experts_2bit.py   # 4-bit to 2-bit expert requantization

  paper/
    flash_moe.pdf            # Technical paper with full details
```

## Credits

- Original [flash-moe](https://github.com/danveloper/flash-moe) by Dan Woods (danveloper)
- Inspired by [Karpathy's autoresearch](https://github.com/karpathy) approach
- Apple's ["LLM in a Flash: Efficient Large Language Model Inference with Limited Memory"](https://arxiv.org/abs/2312.11514) (Alizadeh et al., 2023)
- Models from the [MLX Community](https://huggingface.co/mlx-community) on HuggingFace
