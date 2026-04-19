# Speculative Decoding for flash-moe — Implementation Plan

## Concept

Integrate DFlash-style speculative decoding directly into the flash-moe C/Metal engine,
without requiring Python/MLX at runtime.

## How It Works

```
┌──────────────────────────────────────────────┐
│  DECODE LOOP (current: 1 token per iteration) │
│                                               │
│  NEW:                                         │
│  1. Draft phase: small model generates 16     │
│     candidate tokens (cheap, fast)            │
│  2. Verify phase: target model runs 1 forward │
│     pass over all 16 tokens (batched)         │
│  3. Accept: greedy match prefix, reject rest  │
│  4. Rollback: restore KV/GDN cache to         │
│     accepted position                         │
└──────────────────────────────────────────────┘
```

## What Needs To Be Built

### 1. Draft Model Engine (NEW — the big piece)

A small ~1B param model that runs alongside flash-moe and generates 16 token candidates.

**Options:**

#### Option A: Load DFlash draft model from HuggingFace (RECOMMENDED)
- Draft model `z-lab/Qwen3.6-35B-A3B-DFlash` is ~474MB
- Convert to flash-moe binary format (model_weights.bin + packed experts)
- It's a standard transformer with cross-attention to target hidden states
- Need to implement:
  - Draft model weight loading (mmap)
  - Draft forward pass (Metal shaders for projections, attention)
  - Cross-attention mechanism (draft attends to target hidden states)

#### Option B: Tiny fallback — n-gram / repetition predictor
- Zero-cost draft: predict next tokens from recent token history
- Much lower acceptance rate (~60% vs ~90%) but zero extra RAM
- Good as fallback when no draft model loaded

### 2. Batched Verify Pass (MODIFY existing code)

Currently `fused_layer_forward()` processes 1 token at a time (hidden[HIDDEN_DIM]).

For verify, need to process 16 tokens at once:
- `hidden[16][HIDDEN_DIM]` → batch through all layers
- KV cache: append 16 positions at once (already supported during prefill)
- GDN state: need tape-replay rollback (new)
- Output: `logits[16][VOCAB_SIZE]` → argmax each → compare with draft tokens

**Key changes to infer.m:**
```
Line 6750: for (int gen = 0; gen < max_gen; gen++) {
    // Currently: 1 token per iteration
    
    // NEW speculative loop:
    // if draft model loaded:
    //   1. draft_forward() → draft_tokens[16]
    //   2. batched_verify_forward(draft_tokens, 16) → verify_logits[16][VOCAB_SIZE]
    //   3. match_acceptance() → n_accepted
    //   4. rollback_cache(n_accepted)
    // else:
    //   existing single-token path (unchanged)
```

### 3. Cache Rollback (NEW)

After verify, if only N of 16 tokens are accepted, need to rollback:

**Full attention layers (KV cache):**
- Easy: just truncate KV cache length to `position + n_accepted`
- KV entries beyond that point are discarded

**Linear attention layers (GatedDeltaNet state):**
- Hard: GDN state is a recurrent matrix that's updated per-token
- Options:
  a. **Snapshot + restore**: save state before verify, restore and replay N steps
  b. **Innovation tape** (dflash approach): record deltas during verify, replay only accepted
  - Recommended: approach (b) — lower memory, matches dflash proven technique
  - Need new Metal kernel: `tape_replay()` (see dflash-mlx kernels.py)

### 4. Draft Model Cross-Attention (NEW Metal shaders)

Draft model needs access to target model's hidden states at specific layers.

```
DFlash draft model architecture:
- Small transformer (~6 layers, hidden=1024)
- Each layer has:
  1. Self-attention (standard, on draft hidden states)
  2. Cross-attention (draft queries attend to target hidden states)
  3. MLP (standard)
- Target hidden states extracted at evenly-spaced layers
- Projected and concatenated as context for cross-attention
```

New Metal shaders needed:
- `draft_cross_attention` — cross-attention between draft queries and target keys/values
- `draft_projection` — project target hidden to draft dimension

## Implementation Phases

### Phase 1: Batched Verify ( foundation )
**Effort: ~1-2 weeks**

1. Modify `fused_layer_forward()` to accept `int num_tokens` parameter
2. Batch embed lookup, layer forward, lm_head for N tokens
3. KV cache: batch append (prefill path already does this)
4. Test: verify produces identical logits to sequential single-token

### Phase 2: Cache Rollback
**Effort: ~1 week**

1. Implement KV cache truncation (simple)
2. Implement GDN state snapshot/restore
3. Implement innovation tape recording + replay Metal kernel
4. Test: rollback produces identical state to sequential processing

### Phase 3: Draft Model Loading & Forward Pass
**Effort: ~2-3 weeks**

1. Python script: convert DFlash draft model (safetensors → binary format)
2. C code: mmap draft model weights
3. Metal shaders: draft self-attention, MLP, cross-attention
4. C code: draft forward pass orchestration
5. Test: draft model outputs match Python reference

### Phase 4: Integration & Testing
**Effort: ~1 week**

1. Wire draft → verify → accept/rollback into main decode loop
2. CLI flag: `--draft-model path/to/draft.bin`
3. Fallback: if no draft model, use existing single-token path
4. Benchmark: measure acceptance rate, tok/s improvement
5. Server mode: integrate with OpenAI-compatible API

## Estimated Speedup

Based on dflash-mlx benchmarks adapted for flash-moe:

| Scenario | Baseline | Expected DFlash | Speedup |
|---|---|---|---|
| M4 Pro, 35B-A3B, 8-bit | ~21 tok/s | ~30-35 tok/s | **1.4-1.7x** |
| M4 Pro, 35B-A3B, ctx 1024 | ~21 tok/s | ~33-38 tok/s | **1.5-1.8x** |
| M4 Pro, 35B-A3B, ctx 8192 | ~18 tok/s | ~22-25 tok/s | **1.2-1.4x** |

Lower than dflash-mlx because:
- SSD streaming bottleneck (expert pread) limits batch verify speedup
- C/Metal implementation may not have MLX's optimized batched kernels
- But: no Python overhead, no GIL, pure native performance

## RAM Impact

| Component | Additional RAM |
|---|---|
| Draft model weights (mmap) | ~0.5 GB |
| Draft KV cache (per request) | ~20 MB per 1024 tokens |
| Innovation tape (per verify cycle) | ~2 MB |
| Target hidden state capture | ~0.5 MB |
| **Total additional** | **~0.5-1 GB** |

Fits within M4 Pro 24GB with the 35B-A3B model.

## Files to Create/Modify

### New files:
- `metal_infer/draft_infer.m` — draft model forward pass orchestration
- `metal_infer/draft_shaders.metal` — Metal shaders for draft model
- `metal_infer/rollback.m` — cache rollback logic (KV + GDN)
- `metal_infer/verify_batch.m` — batched verify forward pass
- `convert_draft_model.py` — convert DFlash safetensors to flash-moe binary

### Modified files:
- `metal_infer/infer.m` — main decode loop, add speculative path
- `metal_infer/main.m` — CLI args for --draft-model
- `metal_infer/shaders.metal` — add batched matmul shaders
- `Makefile` — compile new files

## Open Questions

1. **Draft model format**: Should we keep same binary format as target, or simpler (no MoE)?
   → Draft model is NOT MoE, it's a dense transformer. Simpler format.
   
2. **Context 65K-128K**: Speculative decoding alone won't solve this.
   Need separate work: KV cache offloading, sliding window, RoPE scaling.
   See `docs/long-context-research.md` (TODO).

3. **Multiple concurrent requests**: Draft model state per-request vs shared?
   → Draft model weights shared (mmap), KV cache per-request.
   Need per-request draft state management in server mode.
