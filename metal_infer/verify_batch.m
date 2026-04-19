/*
 * verify_batch.m — Batched verify forward pass for speculative decoding
 *
 * Takes N draft tokens, runs them through the model sequentially (like prefill),
 * and produces per-token logits for greedy acceptance matching.
 *
 * This is Phase 1: functional correctness. Later phases will:
 * - Add cache snapshot + rollback for partial acceptance
 * - Optimize with batched GPU kernels
 * - Integrate draft model for speculative generation
 *
 * Build: compiled as part of infer.m (included via Makefile)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations from infer.m
// These are defined in infer.m and accessible when linked together.

// Max tokens we'll verify in one batch call
#define VERIFY_MAX_TOKENS 16

// ============================================================================
// verify_batch_forward — Run N tokens through model, produce per-token logits
// ============================================================================
//
// This function processes N tokens one-by-one through the full model stack
// (embedding → layers → norm → lm_head), exactly like the decode loop does.
// The key difference: it saves hidden states and logits at each step so we
// can later match against draft tokens.
//
// Parameters:
//   wf           — weight file (model weights, mmap'd)
//   token_ids    — array of N token IDs to verify
//   N            — number of tokens (1..VERIFY_MAX_TOKENS)
//   start_pos    — starting position in the sequence (for KV cache/RoPE)
//   hidden       — working buffer [HIDDEN_DIM], must be preserved between calls
//   kv_caches    — KV caches for full attention layers [NUM_LAYERS]
//   layer_states — GDN states for linear attention layers [NUM_LAYERS] (GPU buffers)
//   layer_mmaps  — mmap'd expert files per layer [NUM_LAYERS]
//   layer_fds    — file descriptors for packed expert files [NUM_LAYERS]
//   K            — number of active experts per token
//   logits_out   — output: [N][VOCAB_SIZE] per-token logits (caller-allocated)
//   final_norm_w — final RMS norm weights (or NULL if not loaded)
//
// Returns: 0 on success, -1 on error
//
// After calling this:
//   1. Compare logits_out[i] argmax with token_ids[i] for greedy acceptance
//   2. Count accepted tokens (matching prefix)
//   3. Rollback caches to accepted position (Phase 2)
//
// NOTE: This function modifies kv_caches and layer_states in place.
//       Caller must handle rollback if partial acceptance.
// ============================================================================

static int verify_batch_forward(
    void *wf,                    // WeightFile* — void to avoid header coupling
    const int *token_ids,        // [N] draft token IDs to verify
    int N,                       // number of tokens
    int start_pos,               // starting position
    float *hidden,               // [HIDDEN_DIM] working buffer
    void **kv_caches,            // [NUM_LAYERS] KVCache* array
    void **layer_states,         // [NUM_LAYERS] id<MTLBuffer>* array (GPU delta states)
    void **layer_mmaps,          // [NUM_LAYERS] mmap bases
    int *layer_fds,              // [NUM_LAYERS] packed expert file descriptors
    int K,                       // active experts
    float *logits_out,           // [N * VOCAB_SIZE] output logits
    const float *final_norm_w    // [HIDDEN_DIM] or NULL
) {
    if (N <= 0 || N > VERIFY_MAX_TOKENS) {
        fprintf(stderr, "[verify_batch] invalid N=%d (max %d)\n", N, VERIFY_MAX_TOKENS);
        return -1;
    }

    // We need to cast back to proper types.
    // Since infer.m defines these types, we use the actual function signatures.
    // The linker will resolve everything.
    
    // This file will be #include'd in infer.m or compiled separately.
    // For now, we declare extern references to functions defined in infer.m.

    // EXTERN FUNCTIONS (defined in infer.m):
    // extern void embed_lookup(WeightFile *wf, int token_id, float *out);
    // extern void fused_layer_forward(WeightFile*, int, float*, KVCache*, 
    //                                  id<MTLBuffer>, int, const void*, int, int);
    // extern void lm_head_forward(WeightFile *wf, const float *hidden, float *logits);
    // extern void cpu_rms_norm(const float *x, const float *w, float *out, int dim, float eps);
    // extern void suppress_think_token(float *logits);
    // extern void complete_deferred_experts(void);
    // extern void discard_deferred_experts(void);

    // Since we can't easily call these without proper type declarations,
    // Phase 1 will be implemented directly in infer.m as a new function
    // that has access to all the static declarations.
    
    // See the actual implementation in infer.m: verify_batch_forward_impl()
    fprintf(stderr, "[verify_batch] stub — use verify_batch_forward_impl() in infer.m\n");
    return -1;
}

// ============================================================================
// Greedy acceptance matching
// ============================================================================
//
// Compare draft tokens with verified logits using greedy argmax.
// Returns the number of accepted tokens (matching prefix length).
//
// accepted[0..N-1] is set to 1 for accepted, 0 for rejected.
// Rejection is always at the first mismatch — all tokens after are rejected.

static int verify_acceptance(
    const int *draft_tokens,     // [N] what the draft model predicted
    const float *logits,         // [N * VOCAB_SIZE] verified logits
    int N,                       // number of tokens
    int *accepted                // [N] output: 1=accepted, 0=rejected (optional, can be NULL)
) {
    int n_accepted = 0;
    
    for (int i = 0; i < N; i++) {
        const float *token_logits = logits + (size_t)i * VOCAB_SIZE;
        
        // Greedy argmax
        int best = 0;
        float best_val = token_logits[0];
        for (int v = 1; v < VOCAB_SIZE; v++) {
            if (token_logits[v] > best_val) {
                best_val = token_logits[v];
                best = v;
            }
        }
        
        if (best == draft_tokens[i]) {
            n_accepted++;
            if (accepted) accepted[i] = 1;
        } else {
            // First mismatch — reject this and all remaining
            if (accepted) {
                for (int j = i; j < N; j++) accepted[j] = 0;
            }
            break;
        }
    }
    
    return n_accepted;
}

// ============================================================================
// KV Cache rollback — truncate full attention layer caches
// ============================================================================

// This function will be called with proper KVCache* types from infer.m.
// Provided here as documentation of the rollback logic:
//
// For each full attention layer:
//   kv_cache->len = original_len + n_accepted;
//   (entries beyond that are simply ignored on next use)
//
// For GDN linear attention layers:
//   Need innovation tape replay — Phase 2.

static int verify_rollback_count = 0;  // stats tracking
static int verify_total_accepted = 0;  // stats tracking
