# DFlash Speculative Decoding Integration

## Apa itu DFlash?

DFlash (arXiv:2602.06036, Chen et al. 2026) adalah teknik **speculative decoding** untuk Apple Silicon yang bisa meningkatkan token/s **1.2x - 4.4x** tanpa mengurangi kualitas output (lossless).

**Cara kerja:**
1. Model kecil (~474MB draft model) generate **16 token sekaligus** (block diffusion)
2. Model target (flash-moe) **verifikasi 16 token dalam 1 forward pass**
3. Token yang benar diterima, yang salah ditolak — **output identik** dengan autoregressive biasa

## Benchmark (M5 Max 64GB, dari repo dflash-mlx)

| Target Model | Baseline | DFlash | Speedup | Acceptance |
|---|---|---|---|---|
| Qwen3.5-35B-A3B-4bit | 141 tok/s | 255 tok/s | **1.8x** | 89.7% |
| Qwen3.6-35B-A3B-4bit | 139 tok/s | 253 tok/s | **1.8x** | 89.6% |
| Qwen3.5-9B | 31 tok/s | 113 tok/s | **3.7x** | 89.2% |

> ⚠️ Benchmark di atas di M5 Max 64GB. Di M4 Pro 24GB speedup kemungkinan lebih rendah tapi tetap signifikan.

## Setup (di atas flash-moe yang sudah ada)

### 1. Install dflash-mlx

```bash
python3.12 -m venv ~/dflash-venv
source ~/dflash-venv/bin/activate
pip install dflash-mlx
```

### 2. Request akses draft model

Draft model DFlash adalah **gated repo** — perlu manual request akses:

- **Qwen3.6-35B-A3B:** https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash
- **Qwen3.5-35B-A3B:** https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash

Klik "Agree and access repository" → tunggu approval (biasanya instan).

### 3. Test generate

```bash
source ~/dflash-venv/bin/activate

# Dengan model 8-bit yang sudah ada di cache (flash-moe)
dflash --model mlx-community/Qwen3.6-35B-A3B-8bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --prompt "Test speculative decoding" --max-tokens 100
```

### 4. Serve sebagai OpenAI-compatible server (port terpisah dari flash-moe)

```bash
source ~/dflash-venv/bin/activate

dflash-serve --model mlx-community/Qwen3.6-35B-A3B-8bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --port 8082 \
  --chat-template-args '{"enable_thinking": false}'
```

## Arsitektur Stack

```
┌──────────────────────────────┐
│  dflash-mlx (:8082)         │  ← Speculative decoding layer
│  Draft: ~474MB RAM           │  ← 16 token paralel → verify
├──────────────────────────────┤
│  flash-moe (:8081)           │  ← Target model (C/Metal SSD streaming)
│  Target: Qwen3.6-35B-A3B    │  ← 8-bit, tetap jalan seperti biasa
└──────────────────────────────┘
```

**Tidak ada perubahan pada flash-moe.** DFlash jalan sebagai layer terpisah di atas.

## RAM Usage (estimasi M4 Pro 24GB)

| Komponen | RAM |
|---|---|
| Target model 8-bit (resident) | ~18 GB |
| Draft model | ~0.5 GB |
| KV cache + overhead | ~1-2 GB |
| **Total** | **~20-20.5 GB** ✅ muat |

## Context Length & Acceptance Rate

| Context Length | Acceptance Rate | Speedup |
|---|---|---|
| 1024 tokens | ~91% | **2.2x** |
| 2048 tokens | ~90% | **1.8x** |
| 4096 tokens | ~88% | **1.6x** |
| 8192 tokens | ~87% | **1.3x** |
| 8192+ tokens | turun drastis | butuh riset |

### Riset: Context 65K-128K

**Masalah:** Acceptance rate speculative decoding turun di context panjang karena:
1. Draft model kecil tidak punya kapasitas "memahami" context panjang
2. KV cache draft model tidak scalable
3. Verify step menjadi bottleneck (attention O(n²))

**Teknik yang sedang dieksplorasi:**
- **Sliding window + sink tokens** untuk draft KV cache (dflash sudah implement: `DFLASH_DRAFT_SINK=64`, `DFLASH_DRAFT_WINDOW=1024`)
- **RoPE scaling / YaRN** untuk extend positional encoding
- **Chunked prefill** untuk menghindari OOM
- **KV cache quantization** (flash-moe sudah punya `--kv-quant`)
- **Draft model distillation** yang lebih akurat di long context

**Environment variables untuk tuning:**
```bash
DFLASH_DRAFT_SINK=64        # Sink tokens di awal context
DFLASH_DRAFT_WINDOW=1024    # Sliding window size
DFLASH_MAX_CTX=16384        # Max context sebelum fallback ke AR biasa
DFLASH_PROFILE=1            # Enable profiling
```

## File-file penting dflash-mlx

| File | Fungsi |
|---|---|
| `dflash_mlx/runtime.py` | Core runtime: stream_dflash_generate, verify, rollback |
| `dflash_mlx/model.py` | DFlashDraftModel (draft network dengan cross-attention ke target) |
| `dflash_mlx/kernels.py` | Metal kernels: gated_delta_tape, tape_replay, SDPA 2-pass |
| `dflash_mlx/recurrent_rollback_cache.py` | RecurrentRollbackCache untuk GDN state coherence |
| `dflash_mlx/draft_backend.py` | EagerDraftBackend: draft greedy generation |
| `dflash_mlx/verify_qmm.py` | Custom M=16 quantized matmul Metal kernel |
| `dflash_mlx/verify_linear.py` | VerifyQuantizedLinear wrapper |
| `dflash_mlx/serve.py` | OpenAI-compatible server |

## Limitasi

- dflash-mlx v0.1.0 (pip) belum include verify_qmm/verify_linear (butuh v0.1.4.1 dari git)
- Model 122B **tidak feasible** untuk dflash — butuh RAM terlalu besar
- Draft model gated — perlu manual approval HuggingFace
- Context >8K speedup menurun signifikan
