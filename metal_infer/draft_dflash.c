/*
 * draft_dflash.c — DFlash draft model forward pass
 *
 * Produces 15 draft tokens per call using a small 8-layer transformer
 * conditioned on target model hidden states (cross-self hybrid attention).
 *
 * Architecture:
 *   - 8 standard transformer layers (no MoE)
 *   - hidden_dim=2048, num_heads=32 (Q), num_kv_heads=4, head_dim=128
 *   - FFN intermediate=6144 (SwiGLU)
 *   - Cross-self hybrid attention: shared k_proj/v_proj for context + noise
 *   - QK-norm: per-head RMSNorm on Q and K
 *   - fc: [2048, 10240] projects concatenated target hidden states → 2048
 *   - hidden_norm: RMSNorm after fc
 *   - block_size=16 → produces 15 draft tokens
 *   - RoPE with YaRN scaling (factor=64, theta=10M)
 *
 * Designed to be #included from infer.m (shares cblas_sgemm, embed_lookup,
 * lm_head_forward, cpu_rms_norm, cpu_argmax, etc.)
 */

#ifdef DFLASH_IMPL

// ============================================================================
// Constants
// ============================================================================

#define DFLASH_NUM_LAYERS       8
#define DFLASH_HIDDEN_DIM       2048
#define DFLASH_NUM_HEADS        32
#define DFLASH_NUM_KV_HEADS     4
#define DFLASH_HEAD_DIM         128
#define DFLASH_INTERMEDIATE     6144
#define DFLASH_BLOCK_SIZE       16
#define DFLASH_NUM_DRAFT_TOKENS (DFLASH_BLOCK_SIZE - 1)  // 15
#define DFLASH_TARGET_LAYERS    5
#define DFLASH_TARGET_CONCAT    (DFLASH_TARGET_LAYERS * DFLASH_HIDDEN_DIM)  // 10240
#define DFLASH_MASK_TOKEN_ID    248070
#define DFLASH_ROPE_THETA       10000000.0f
#define DFLASH_YARN_FACTOR      64.0f
#define DFLASH_RMS_EPS          1e-6f

// ============================================================================
// Weight offsets — computed from draft_weights_meta.json at init time
// ============================================================================

typedef struct {
    // Global weights
    float *fc_weight;           // [2048, 10240]
    float *hidden_norm_weight;  // [2048]
    float *final_norm_weight;   // [2048]

    // Per-layer weights (array of DFLASH_NUM_LAYERS)
    struct {
        float *input_layernorm;     // [2048]
        float *q_proj;             // [4096, 2048]
        float *k_proj;             // [512, 2048]
        float *v_proj;             // [512, 2048]
        float *o_proj;             // [2048, 4096]
        float *q_norm;             // [128]
        float *k_norm;             // [128]
        float *post_attn_layernorm; // [2048]
        float *gate_proj;          // [6144, 2048]
        float *up_proj;            // [6144, 2048]
        float *down_proj;          // [2048, 6144]
    } layers[DFLASH_NUM_LAYERS];
} DFlashWeights;

// ============================================================================
// Persistent state
// ============================================================================

static DFlashWeights dflash_w;
static float *dflash_data = NULL;       // mmap'd weight data
static size_t dflash_data_size = 0;
static int dflash_initialized = 0;

// Scratch buffers (allocated once)
static float *df_noise_emb = NULL;      // [BLOCK_SIZE, HIDDEN_DIM]
static float *df_projected = NULL;      // [seq_len, HIDDEN_DIM] — projected target hidden
static float *df_projected_normed = NULL; // [seq_len, HIDDEN_DIM] — after rms_norm before hidden_norm
static float *df_hidden = NULL;         // [BLOCK_SIZE, HIDDEN_DIM] — current layer hidden state
static float *df_residual = NULL;       // [BLOCK_SIZE, HIDDEN_DIM]
static float *df_normed = NULL;         // [BLOCK_SIZE, HIDDEN_DIM]
static float *df_q = NULL;              // [BLOCK_SIZE, NUM_HEADS * HEAD_DIM]
static float *df_k_noise = NULL;        // [BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM]
static float *df_v_noise = NULL;        // [BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM]
static float *df_k_ctx = NULL;          // [seq_len, NUM_KV_HEADS * HEAD_DIM]
static float *df_v_ctx = NULL;          // [seq_len, NUM_KV_HEADS * HEAD_DIM]
static float *df_k_all = NULL;          // [seq_len + BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM]
static float *df_v_all = NULL;          // [seq_len + BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM]
static float *df_attn_scores = NULL;    // [BLOCK_SIZE * NUM_HEADS, seq_len + BLOCK_SIZE]
static float *df_attn_out = NULL;       // [BLOCK_SIZE, NUM_HEADS * HEAD_DIM]
static float *df_o_proj = NULL;         // [BLOCK_SIZE, HIDDEN_DIM]
static float *df_gate = NULL;           // [BLOCK_SIZE, INTERMEDIATE]
static float *df_up = NULL;             // [BLOCK_SIZE, INTERMEDIATE]
static float *df_down = NULL;           // [BLOCK_SIZE, HIDDEN_DIM]
static float *df_logits = NULL;         // [BLOCK_SIZE, VOCAB_SIZE] — only need DRAFT_TOKENS rows
static float *df_scratch = NULL;        // [BLOCK_SIZE, HIDDEN_DIM] — temp scratch

// YaRN RoPE: precomputed angular frequencies and mscale
static float df_yarn_freqs[64];         // [half_dim] angular frequencies
static float df_yarn_mscale = 1.0f;     // mscale ratio for YaRN

// Max context length for projected target hidden states
#define DFLASH_MAX_SEQ 4096

// ============================================================================
// Helpers
// ============================================================================

// RMSNorm with float32 weights (draft weights are stored as FP32)
static void dflash_rms_norm(float *out, const float *x, const float *w, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms * w[i];
}

// Bare RMSNorm (no weight — just normalize)
static void dflash_rms_norm_bare(float *out, const float *x, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms;
}

// Per-head RMSNorm: apply RMSNorm independently to each head's vector
static void dflash_per_head_norm(float *qk, int num_heads, int head_dim, const float *w) {
    for (int h = 0; h < num_heads; h++) {
        dflash_rms_norm(qk + h * head_dim, qk + h * head_dim, w, head_dim, DFLASH_RMS_EPS);
    }
}

// ============================================================================
// YaRN RoPE: compute correct frequencies at init time
// ============================================================================

static void dflash_compute_yarn_freqs(void) {
    int dims = DFLASH_HEAD_DIM;     // 128
    int half_dims = dims / 2;       // 64
    float base = DFLASH_ROPE_THETA; // 10000000.0f
    float scaling_factor = DFLASH_YARN_FACTOR; // 64.0f
    float original_max_pos = 4096.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;

    // Compute correction range
    int low = (int)floorf(
        (dims * logf(original_max_pos / (beta_fast * 2.0f * (float)M_PI))) / (2.0f * logf(base))
    );
    int high = (int)ceilf(
        (dims * logf(original_max_pos / (beta_slow * 2.0f * (float)M_PI))) / (2.0f * logf(base))
    );

    for (int d = 0; d < half_dims; d++) {
        float freq_extra = powf(base, 2.0f * d / dims);
        float freq_inter = scaling_factor * freq_extra;

        // Linear ramp mask
        float ramp;
        if (low == high) ramp = 0.0f;
        else ramp = fmaxf(0.0f, fminf(1.0f, (float)(d - low) / (float)(high - low)));
        float mask = 1.0f - ramp;

        float yarn_freq = (freq_inter * freq_extra) / (freq_inter * mask + freq_extra * (1.0f - mask));
        df_yarn_freqs[d] = 1.0f / yarn_freq;
    }

    // mscale from DFlash reference (model.py yarn_get_mscale)
    // mscale = 0.1 * mscale_param * log(factor) + 1.0, with mscale_param=1
    // From mlx_lm YarnRoPE: yarn_get_mscale(factor, mscale=1) / yarn_get_mscale(factor, mscale_all_dim=0)
    // = (0.1 * 1 * log(factor) + 1.0) / (0.1 * 0 * log(factor) + 1.0)
    // = 0.1 * log(factor) + 1.0
    df_yarn_mscale = 0.1f * logf((float)scaling_factor) + 1.0f;

}

// ============================================================================
// RoPE with YaRN scaling (applied in-place)
//
// Uses precomputed YaRN frequencies with correction range and ramp mask.
// Applies mscale scaling to the first half_dim before rotation.
// ============================================================================

static void dflash_apply_rope(float *x, int seq_len, int num_heads, int head_dim, int start_pos) {
    int half_dim = head_dim / 2;  // 64

    for (int s = 0; s < seq_len; s++) {
        int pos = start_pos + s;
        for (int h = 0; h < num_heads; h++) {
            float *head = x + (s * num_heads + h) * head_dim;

            for (int d = 0; d < half_dim; d++) {
                float angle = pos * df_yarn_freqs[d];
                float cos_a = cosf(angle) * df_yarn_mscale;
                float sin_a = sinf(angle) * df_yarn_mscale;

                float x0 = head[d];
                float x1 = head[d + half_dim];
                head[d]          = x0 * cos_a - x1 * sin_a;
                head[d + half_dim] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

// ============================================================================
// Init: mmap draft weights, compute pointer offsets, allocate scratch
// ============================================================================

static int dflash_init(void) {
    if (dflash_initialized) return 0;

    // ---- mmap the weight file ----
    const char *bin_path = "draft_weights.bin";
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[dflash] Cannot open %s: %s\n", bin_path, strerror(errno));
        return -1;
    }

    struct stat st;
    fstat(fd, &st);
    dflash_data_size = st.st_size;
    dflash_data = mmap(NULL, dflash_data_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (dflash_data == MAP_FAILED) {
        fprintf(stderr, "[dflash] mmap failed: %s\n", strerror(errno));
        return -1;
    }

    fprintf(stderr, "[dflash] mmap'd %s: %.1f MB\n", bin_path, dflash_data_size / 1e6);

    // ---- Parse metadata JSON to find offsets ----
    // We hardcode the offsets based on draft_weights_meta.json since it's fixed.
    // This avoids pulling in a JSON parser. The offsets are deterministic from
    // convert_draft_weights.py (sorted by tensor name, FP32).
    //
    // Offset table (from draft_weights_meta.json):
    //   fc.weight:                  0
    //   hidden_norm.weight:         83886080
    //   layers.0.input_layernorm:   83894272
    //   layers.0.mlp.down_proj:     83902464
    //   layers.0.mlp.gate_proj:     134234112
    //   layers.0.mlp.up_proj:       184565760
    //   layers.0.post_attn_ln:      234897408
    //   layers.0.k_norm:            234905600
    //   layers.0.k_proj:            234906112
    //   layers.0.o_proj:            239100416
    //   layers.0.q_norm:            272654848
    //   layers.0.q_proj:            272655360
    //   layers.0.v_proj:            306209792
    //   ... per-layer stride = 226492416 bytes

    #define DFLASH_OFF_FC                0
    #define DFLASH_OFF_HIDDEN_NORM       83886080
    #define DFLASH_OFF_NORM              1895972864  // final norm

    // Layer stride (offsets between consecutive layers, determined from meta JSON)
    // Layer 0 input_layernorm offset:  83894272
    // Layer 1 input_layernorm offset:  310404096
    // Stride = 310404096 - 83894272 = 226509824
    #define DFLASH_LAYER_STRIDE         226509824

    // Offsets within a layer (relative to layer's input_layernorm)
    //   input_layernorm:       +0
    //   mlp.down_proj:         +8192    (83894272 → 83902464)
    //   mlp.gate_proj:         +50331648+8192 = +50339840 (134234112 - 83894272)
    //   mlp.up_proj:           +100663296+8192 = +100671488 (184565760 - 83894272)
    //   post_attn_layernorm:   +150994944+8192 = +151003136 (234897408 - 83894272)
    //   k_norm:                +150994944+8192+8192 = +151011328 (234905600 - 83894272)
    //   k_proj:                +150994944+8192+8192+512 = +151011840 (234906112 - 83894272)
    //   o_proj:                +155189344 (239100416 - 83894272)
    //   q_norm:                +188760576 (272654848 - 83894272)
    //   q_proj:                +188761088 (272655360 - 83894272)
    //   v_proj:                +222315520 (306209792 - 83894272)

    #define DFLASH_LN_OFF_INPUT_NORM    0
    #define DFLASH_LN_OFF_DOWN_PROJ     8192
    #define DFLASH_LN_OFF_GATE_PROJ     50339840
    #define DFLASH_LN_OFF_UP_PROJ       100671488
    #define DFLASH_LN_OFF_POST_ATTN     151003136
    #define DFLASH_LN_OFF_K_NORM        151011328
    #define DFLASH_LN_OFF_K_PROJ        151011840
    #define DFLASH_LN_OFF_O_PROJ        155206144
    #define DFLASH_LN_OFF_Q_NORM        188760576
    #define DFLASH_LN_OFF_Q_PROJ        188761088
    #define DFLASH_LN_OFF_V_PROJ        222315520

    // ---- Set weight pointers ----
    char *base = (char *)dflash_data;

    dflash_w.fc_weight          = (float *)(base + DFLASH_OFF_FC);
    dflash_w.hidden_norm_weight = (float *)(base + DFLASH_OFF_HIDDEN_NORM);
    dflash_w.final_norm_weight  = (float *)(base + DFLASH_OFF_NORM);

    size_t layer0_base = 83894272;  // absolute offset of layer 0 input_layernorm
    for (int l = 0; l < DFLASH_NUM_LAYERS; l++) {
        char *lb = base + layer0_base + (size_t)l * DFLASH_LAYER_STRIDE;
        dflash_w.layers[l].input_layernorm     = (float *)(lb + DFLASH_LN_OFF_INPUT_NORM);
        dflash_w.layers[l].down_proj           = (float *)(lb + DFLASH_LN_OFF_DOWN_PROJ);
        dflash_w.layers[l].gate_proj           = (float *)(lb + DFLASH_LN_OFF_GATE_PROJ);
        dflash_w.layers[l].up_proj             = (float *)(lb + DFLASH_LN_OFF_UP_PROJ);
        dflash_w.layers[l].post_attn_layernorm = (float *)(lb + DFLASH_LN_OFF_POST_ATTN);
        dflash_w.layers[l].k_norm              = (float *)(lb + DFLASH_LN_OFF_K_NORM);
        dflash_w.layers[l].k_proj              = (float *)(lb + DFLASH_LN_OFF_K_PROJ);
        dflash_w.layers[l].o_proj              = (float *)(lb + DFLASH_LN_OFF_O_PROJ);
        dflash_w.layers[l].q_norm              = (float *)(lb + DFLASH_LN_OFF_Q_NORM);
        dflash_w.layers[l].q_proj              = (float *)(lb + DFLASH_LN_OFF_Q_PROJ);
        dflash_w.layers[l].v_proj              = (float *)(lb + DFLASH_LN_OFF_V_PROJ);
    }

    // ---- Allocate scratch buffers ----
    int B = DFLASH_BLOCK_SIZE;
    int HD = DFLASH_HIDDEN_DIM;
    int QD = DFLASH_NUM_HEADS * DFLASH_HEAD_DIM;     // 4096
    int KVD = DFLASH_NUM_KV_HEADS * DFLASH_HEAD_DIM;  // 512
    int INT = DFLASH_INTERMEDIATE;                     // 6144
    int MS = DFLASH_MAX_SEQ;

    df_noise_emb       = malloc((size_t)B * HD * sizeof(float));
    df_projected       = malloc((size_t)MS * HD * sizeof(float));
    df_projected_normed= malloc((size_t)MS * HD * sizeof(float));
    df_hidden          = malloc((size_t)B * HD * sizeof(float));
    df_residual        = malloc((size_t)B * HD * sizeof(float));
    df_normed          = malloc((size_t)B * HD * sizeof(float));
    df_q               = malloc((size_t)B * QD * sizeof(float));
    df_k_noise         = malloc((size_t)B * KVD * sizeof(float));
    df_v_noise         = malloc((size_t)B * KVD * sizeof(float));
    df_k_ctx           = malloc((size_t)MS * KVD * sizeof(float));
    df_v_ctx           = malloc((size_t)MS * KVD * sizeof(float));
    df_k_all           = malloc((size_t)(MS + B) * KVD * sizeof(float));
    df_v_all           = malloc((size_t)(MS + B) * KVD * sizeof(float));
    df_attn_scores     = malloc((size_t)B * DFLASH_NUM_HEADS * (MS + B) * sizeof(float));
    df_attn_out        = malloc((size_t)B * QD * sizeof(float));
    df_o_proj          = malloc((size_t)B * HD * sizeof(float));
    df_gate            = malloc((size_t)B * INT * sizeof(float));
    df_up              = malloc((size_t)B * INT * sizeof(float));
    df_down            = malloc((size_t)B * HD * sizeof(float));
    df_logits          = malloc((size_t)B * VOCAB_SIZE * sizeof(float));
    df_scratch         = malloc((size_t)B * HD * sizeof(float));

    if (!df_noise_emb || !df_projected || !df_hidden || !df_residual ||
        !df_normed || !df_q || !df_k_noise || !df_v_noise ||
        !df_k_ctx || !df_v_ctx || !df_k_all || !df_v_all ||
        !df_attn_scores || !df_attn_out || !df_o_proj ||
        !df_gate || !df_up || !df_down || !df_logits || !df_scratch ||
        !df_projected_normed) {
        fprintf(stderr, "[dflash] Failed to allocate scratch buffers\n");
        return -1;
    }

    fprintf(stderr, "[dflash] Initialized. Scratch buffers allocated.\n");

    // Compute YaRN RoPE frequencies
    dflash_compute_yarn_freqs();
    fprintf(stderr, "[dflash] YaRN freqs computed, mscale=%.4f\n", df_yarn_mscale);

    dflash_initialized = 1;
    return 0;
}

// ============================================================================
// Cleanup
// ============================================================================

static void dflash_cleanup(void) {
    if (!dflash_initialized) return;
    if (dflash_data) { munmap(dflash_data, dflash_data_size); dflash_data = NULL; }
    free(df_noise_emb);       df_noise_emb = NULL;
    free(df_projected);       df_projected = NULL;
    free(df_projected_normed);df_projected_normed = NULL;
    free(df_hidden);          df_hidden = NULL;
    free(df_residual);        df_residual = NULL;
    free(df_normed);          df_normed = NULL;
    free(df_q);               df_q = NULL;
    free(df_k_noise);         df_k_noise = NULL;
    free(df_v_noise);         df_v_noise = NULL;
    free(df_k_ctx);           df_k_ctx = NULL;
    free(df_v_ctx);           df_v_ctx = NULL;
    free(df_k_all);           df_k_all = NULL;
    free(df_v_all);           df_v_all = NULL;
    free(df_attn_scores);     df_attn_scores = NULL;
    free(df_attn_out);        df_attn_out = NULL;
    free(df_o_proj);          df_o_proj = NULL;
    free(df_gate);            df_gate = NULL;
    free(df_up);              df_up = NULL;
    free(df_down);            df_down = NULL;
    free(df_logits);          df_logits = NULL;
    free(df_scratch);         df_scratch = NULL;
    dflash_initialized = 0;
}

// ============================================================================
// Forward pass: produce 15 draft tokens
//
// Arguments:
//   wf                — target model WeightFile (for embed_lookup and lm_head)
//   target_hidden     — concatenated target hidden states [seq_len, 10240]
//                       (5 layers × 2048 dim), row-major FP32
//   seq_len           — number of target positions (prompt + generated so far)
//   staged_first_token — the first draft token (already verified / chosen)
//   current_pos       — current position in the sequence
//   draft_tokens      — output array of 15 ints (caller-allocated)
//   logits_out        — optional output [15 * VOCAB_SIZE] for verification logits (or NULL)
// ============================================================================

int dflash_forward(
    WeightFile *wf,
    const float *target_hidden,     // [seq_len, 10240]
    int seq_len,
    int staged_first_token,
    int current_pos,
    int *draft_tokens,              // output: [15] token IDs
    float *logits_out               // optional output: [15 * VOCAB_SIZE] or NULL
) {
    if (!dflash_initialized) {
        if (dflash_init() != 0) return -1;
    }

    // Window context to last DFLASH_CTX_WINDOW positions to limit computation
    #define DFLASH_CTX_WINDOW 128
    int ctx_offset = 0;
    int effective_seq_len = seq_len;
    if (seq_len > DFLASH_CTX_WINDOW) {
        ctx_offset = seq_len - DFLASH_CTX_WINDOW;
        effective_seq_len = DFLASH_CTX_WINDOW;
    }
    const float *windowed_hidden = target_hidden + (size_t)ctx_offset * DFLASH_TARGET_CONCAT;

    if (seq_len > DFLASH_MAX_SEQ) {
        fprintf(stderr, "[dflash] seq_len=%d exceeds max=%d\n", seq_len, DFLASH_MAX_SEQ);
        return -1;
    }

    int B = DFLASH_BLOCK_SIZE;      // 16
    int HD = DFLASH_HIDDEN_DIM;     // 2048
    int NH = DFLASH_NUM_HEADS;      // 32
    int NKV = DFLASH_NUM_KV_HEADS;  // 4
    int DH = DFLASH_HEAD_DIM;       // 128
    int QD = NH * DH;               // 4096
    int KVD = NKV * DH;             // 512
    int INT = DFLASH_INTERMEDIATE;  // 6144
    float scale = 1.0f / sqrtf((float)DH);  // 1/sqrt(128)

    // ========================================================================
    // Step 1: Embed [staged_first_token, mask×15] using target's embed_tokens
    //         → noise_embedding [16, 2048]
    // ========================================================================
    embed_lookup(wf, staged_first_token, df_noise_emb);
    for (int i = 1; i < B; i++) {
        embed_lookup(wf, DFLASH_MASK_TOKEN_ID, df_noise_emb + (size_t)i * HD);
    }

    // Copy noise embedding into working hidden state
    memcpy(df_hidden, df_noise_emb, (size_t)B * HD * sizeof(float));

    // ========================================================================
    // Step 2: Project target hidden states
    //   projected = hidden_norm(fc(target_hidden_concat))
    //   target_hidden_concat: [seq_len, 10240]
    //   fc: [2048, 10240] × [seq_len, 10240]^T → [seq_len, 2048]
    //   (weights stored row-major: fc_weight[i] = row i = [10240])
    // ========================================================================

    // fc: Y = X @ W^T  where X=[seq_len, 10240], W=[2048, 10240]
    // cblas: C = alpha * A * B + beta * C
    //   A = X [seq_len, 10240], B = W^T [10240, 2048], C = projected [seq_len, 2048]
    {
        // fc_weight is [2048, 10240], stored row-major.
        // We need: projected = X @ W^T
        //   X = target_hidden [seq_len, TC=10240], ldA = TC
        //   W = fc_weight [2048, TC], so W^T is [TC, 2048], ldB = 2048
        // CblasRowMajor, CblasNoTrans, CblasTrans:
        //   C[i,j] = sum_k A[i,k] * B[j,k]  (since B is transposed)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    effective_seq_len, HD, DFLASH_TARGET_CONCAT,
                    1.0f,
                    windowed_hidden, DFLASH_TARGET_CONCAT,
                    dflash_w.fc_weight, DFLASH_TARGET_CONCAT,
                    0.0f,
                    df_projected, HD);
    }

    // Apply hidden_norm (RMSNorm with weights) per position
    for (int s = 0; s < effective_seq_len; s++) {
        float *row = df_projected + (size_t)s * HD;
        dflash_rms_norm(row, row, dflash_w.hidden_norm_weight, HD, DFLASH_RMS_EPS);
    }

    // ========================================================================
    // Step 3: 8 transformer layers
    //   For each layer:
    //     a. residual = hidden
    //     b. hidden = rms_norm(hidden, input_layernorm)
    //     c. Q = q_proj(hidden) [16, 4096] → reshape [16, 32, 128] → RoPE → q_norm
    //     d. context_K = k_proj(projected) [seq_len, 512] → reshape → RoPE → k_norm
    //     e. context_V = v_proj(projected) [seq_len, 512]
    //     f. noise_K = k_proj(hidden) [16, 512] → reshape → RoPE → k_norm
    //     g. noise_V = v_proj(hidden) [16, 512]
    //     h. K = [context_K; noise_K], V = [context_V; noise_V]
    //     i. attn = softmax(Q @ K^T / sqrt(128)) @ V → [16, 4096]
    //     j. hidden = residual + o_proj(attn)
    //     k. residual = hidden
    //     l. hidden = rms_norm(hidden, post_attention_layernorm)
    //     m. gate = silu(gate_proj(hidden)), up = up_proj(hidden)
    //     n. hidden = residual + down_proj(gate * up)
    // ========================================================================

    for (int layer = 0; layer < DFLASH_NUM_LAYERS; layer++) {
        float *w_input_ln  = dflash_w.layers[layer].input_layernorm;
        float *w_q         = dflash_w.layers[layer].q_proj;       // [4096, 2048]
        float *w_k         = dflash_w.layers[layer].k_proj;       // [512, 2048]
        float *w_v         = dflash_w.layers[layer].v_proj;       // [512, 2048]
        float *w_o         = dflash_w.layers[layer].o_proj;       // [2048, 4096]
        float *w_q_norm    = dflash_w.layers[layer].q_norm;       // [128]
        float *w_k_norm    = dflash_w.layers[layer].k_norm;       // [128]
        float *w_post_ln   = dflash_w.layers[layer].post_attn_layernorm;
        float *w_gate      = dflash_w.layers[layer].gate_proj;    // [6144, 2048]
        float *w_up        = dflash_w.layers[layer].up_proj;      // [6144, 2048]
        float *w_down      = dflash_w.layers[layer].down_proj;    // [2048, 6144]

        // ---- 3a: Save residual ----
        memcpy(df_residual, df_hidden, (size_t)B * HD * sizeof(float));

        // ---- 3b: Input layernorm (per position) ----
        for (int s = 0; s < B; s++) {
            dflash_rms_norm(df_normed + (size_t)s * HD,
                            df_hidden + (size_t)s * HD,
                            w_input_ln, HD, DFLASH_RMS_EPS);
        }

        // ---- 3c: Q projection ----
        // Q = df_normed @ w_q^T : [16, 2048] × [2048, 4096] → [16, 4096]
        // w_q is [4096, 2048] row-major. Y = X @ W^T.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B, QD, HD,
                    1.0f,
                    df_normed, HD,
                    w_q, HD,
                    0.0f,
                    df_q, QD);

        // Apply per-head Q-norm BEFORE RoPE (matches Python/MLX reference order)
        for (int s = 0; s < B; s++) {
            float *q_row = df_q + (size_t)s * QD;
            dflash_per_head_norm(q_row, NH, DH, w_q_norm);
        }

        // Apply RoPE to Q: offset = seq_len (full context length, matching Python reference)
        dflash_apply_rope(df_q, B, NH, DH, seq_len);

        // ---- 3d: Context K projection ----
        // context_K = projected @ w_k^T : [seq_len, 2048] × [2048, 512] → [seq_len, 512]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    effective_seq_len, KVD, HD,
                    1.0f,
                    df_projected, HD,
                    w_k, HD,
                    0.0f,
                    df_k_ctx, KVD);

        // Apply per-head K-norm to context K BEFORE RoPE (matches Python/MLX reference)
        for (int s = 0; s < effective_seq_len; s++) {
            float *k_row = df_k_ctx + (size_t)s * KVD;
            dflash_per_head_norm(k_row, NKV, DH, w_k_norm);
        }

        // Apply RoPE to context K: offset = ctx_offset (start of windowed context)
        dflash_apply_rope(df_k_ctx, effective_seq_len, NKV, DH, ctx_offset);

        // ---- 3e: Context V projection ----
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    effective_seq_len, KVD, HD,
                    1.0f,
                    df_projected, HD,
                    w_v, HD,
                    0.0f,
                    df_v_ctx, KVD);

        // ---- 3f: Noise K projection ----
        // noise_K = df_normed @ w_k^T : [16, 2048] × [2048, 512] → [16, 512]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B, KVD, HD,
                    1.0f,
                    df_normed, HD,
                    w_k, HD,
                    0.0f,
                    df_k_noise, KVD);

        // Apply per-head K-norm to noise K BEFORE RoPE (matches Python/MLX reference)
        for (int s = 0; s < B; s++) {
            float *k_row = df_k_noise + (size_t)s * KVD;
            dflash_per_head_norm(k_row, NKV, DH, w_k_norm);
        }

        // Apply RoPE to noise K: offset = seq_len (full context length, matching Python reference)
        dflash_apply_rope(df_k_noise, B, NKV, DH, seq_len);

        // ---- 3g: Noise V projection ----
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B, KVD, HD,
                    1.0f,
                    df_normed, HD,
                    w_v, HD,
                    0.0f,
                    df_v_noise, KVD);

        // ---- 3h: Concatenate K = [context_K; noise_K], V = [context_V; noise_V] ----
        int total_kv_len = effective_seq_len + B;
        memcpy(df_k_all, df_k_ctx, (size_t)effective_seq_len * KVD * sizeof(float));
        memcpy(df_k_all + (size_t)effective_seq_len * KVD, df_k_noise, (size_t)B * KVD * sizeof(float));
        memcpy(df_v_all, df_v_ctx, (size_t)effective_seq_len * KVD * sizeof(float));
        memcpy(df_v_all + (size_t)effective_seq_len * KVD, df_v_noise, (size_t)B * KVD * sizeof(float));

        // ---- 3i: Attention ----
        // Q: [16, 32, 128]  — df_q, row-major with stride QD=4096 between positions
        // K: [total_kv_len, 4, 128] — df_k_all, row-major with stride KVD=512
        // V: [total_kv_len, 4, 128] — df_v_all
        //
        // GQA: 32 Q heads, 4 KV heads → 8 Q heads per KV head
        // For each query position s, for each Q head h:
        //   kv_head = h / 8
        //   score[j] = Q[s,h,:] . K[j,kv_head,:]
        //   After softmax: out[s,h,:] = sum_j score[j] * V[j,kv_head,:]
        {
            int gqa_ratio = NH / NKV;  // 8

            // Compute attention scores and apply attention per-position
            for (int s = 0; s < B; s++) {
                for (int h = 0; h < NH; h++) {
                    int kv_h = h / gqa_ratio;
                    float *q_vec = df_q + (size_t)s * QD + h * DH;
                    float *score_row = df_attn_scores + ((size_t)s * NH + h) * total_kv_len;

                    // Compute Q @ K^T for this head
                    // K layout: df_k_all[pos * KVD + kv_h * DH + d]
                    for (int j = 0; j < total_kv_len; j++) {
                        float dot = 0.0f;
                        const float *k_vec = df_k_all + (size_t)j * KVD + kv_h * DH;
                        for (int d = 0; d < DH; d++) {
                            dot += q_vec[d] * k_vec[d];
                        }
                        score_row[j] = dot * scale;
                    }

                    // Softmax
                    {
                        float max_val = score_row[0];
                        for (int j = 1; j < total_kv_len; j++) {
                            if (score_row[j] > max_val) max_val = score_row[j];
                        }
                        float sum = 0.0f;
                        for (int j = 0; j < total_kv_len; j++) {
                            score_row[j] = expf(score_row[j] - max_val);
                            sum += score_row[j];
                        }
                        float inv_sum = 1.0f / sum;
                        for (int j = 0; j < total_kv_len; j++) {
                            score_row[j] *= inv_sum;
                        }
                    }

                    // Weighted sum: out[s, h, :] = sum_j score[j] * V[j, kv_h, :]
                    float *out_vec = df_attn_out + (size_t)s * QD + h * DH;
                    memset(out_vec, 0, DH * sizeof(float));
                    for (int j = 0; j < total_kv_len; j++) {
                        float weight = score_row[j];
                        const float *v_vec = df_v_all + (size_t)j * KVD + kv_h * DH;
                        for (int d = 0; d < DH; d++) {
                            out_vec[d] += weight * v_vec[d];
                        }
                    }
                }
            }
        }

        // ---- 3j: Output projection + residual ----
        // o_proj: [2048, 4096] — Y = attn_out @ w_o^T : [16, 4096] × [4096, 2048] → [16, 2048]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B, HD, QD,
                    1.0f,
                    df_attn_out, QD,
                    w_o, QD,
                    0.0f,
                    df_o_proj, HD);

        // hidden = residual + o_proj
        for (int i = 0; i < B * HD; i++) {
            df_hidden[i] = df_residual[i] + df_o_proj[i];
        }

        // ---- 3k: Save residual for FFN ----
        memcpy(df_residual, df_hidden, (size_t)B * HD * sizeof(float));

        // ---- 3l: Post-attention layernorm ----
        for (int s = 0; s < B; s++) {
            dflash_rms_norm(df_normed + (size_t)s * HD,
                            df_hidden + (size_t)s * HD,
                            w_post_ln, HD, DFLASH_RMS_EPS);
        }

        // ---- 3m: SwiGLU FFN ----
        // gate = silu(gate_proj(normed))
        // gate_proj: [6144, 2048]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B, INT, HD,
                    1.0f,
                    df_normed, HD,
                    w_gate, HD,
                    0.0f,
                    df_gate, INT);

        // up_proj: [6144, 2048]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B, INT, HD,
                    1.0f,
                    df_normed, HD,
                    w_up, HD,
                    0.0f,
                    df_up, INT);

        // SwiGLU: gate = silu(gate) * up
        for (int i = 0; i < B * INT; i++) {
            float g = df_gate[i];
            float silu_g = g / (1.0f + expf(-g));
            df_gate[i] = silu_g * df_up[i];
        }

        // down_proj: [2048, 6144] — Y = gate @ w_down^T : [16, 6144] × [6144, 2048] → [16, 2048]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B, HD, INT,
                    1.0f,
                    df_gate, INT,
                    w_down, INT,
                    0.0f,
                    df_down, HD);

        // ---- 3n: Residual connection ----
        for (int i = 0; i < B * HD; i++) {
            df_hidden[i] = df_residual[i] + df_down[i];
        }
    }

    // ========================================================================
    // Step 4: Final RMS norm
    // ========================================================================
    for (int s = 0; s < B; s++) {
        float *row = df_hidden + (size_t)s * HD;
        dflash_rms_norm(row, row, dflash_w.final_norm_weight, HD, DFLASH_RMS_EPS);
    }

    // ========================================================================
    // Step 5: Logits via lm_head (skip position 0 → positions 1..15)
    //         logits = lm_head(hidden[1:]) → [15, vocab_size]
    // ========================================================================
    for (int s = 0; s < DFLASH_NUM_DRAFT_TOKENS; s++) {
        float *hidden_row = df_hidden + (size_t)(s + 1) * HD;
        float *logits_row = df_logits + (size_t)s * VOCAB_SIZE;

        // Use the target model's lm_head for logit computation
        lm_head_forward(wf, hidden_row, logits_row);
    }

    // ========================================================================
    // Step 6: Argmax → 15 draft token IDs
    // ========================================================================
    for (int s = 0; s < DFLASH_NUM_DRAFT_TOKENS; s++) {
        float *logits_row = df_logits + (size_t)s * VOCAB_SIZE;
        draft_tokens[s] = cpu_argmax(logits_row, VOCAB_SIZE);
    }

    // Copy logits to optional output buffer
    if (logits_out) {
        memcpy(logits_out, df_logits, (size_t)DFLASH_NUM_DRAFT_TOKENS * VOCAB_SIZE * sizeof(float));
    }

    return 0;
}

#undef DFLASH_NUM_LAYERS
#undef DFLASH_HIDDEN_DIM
#undef DFLASH_NUM_HEADS
#undef DFLASH_NUM_KV_HEADS
#undef DFLASH_HEAD_DIM
#undef DFLASH_INTERMEDIATE
#undef DFLASH_BLOCK_SIZE
#undef DFLASH_NUM_DRAFT_TOKENS
#undef DFLASH_TARGET_LAYERS
#undef DFLASH_TARGET_CONCAT
#undef DFLASH_MASK_TOKEN_ID
#undef DFLASH_ROPE_THETA
#undef DFLASH_YARN_FACTOR
#undef DFLASH_RMS_EPS
#undef DFLASH_MAX_SEQ

#endif // DFLASH_IMPL
