#pragma once
// dit-train.h: LoRA training for the ACE-Step DiT decoder (Phase 2, see
// ../TRAINING_DEV.md). Only the DiT decoder is trained; VAE, text encoder,
// cond encoder and LM stay frozen and are used solely for dataset
// preprocessing (src/dit-train.h has nothing to do with that stage).
//
// Training recipe (matches the Python reference trainer):
//   x0 = target_latents, x1 = noise, t drawn from the turbo shift=3.0
//   discrete schedule, xt = t*x1 + (1-t)*x0, v_pred = DiT(xt, t, t_r=t, ...),
//   loss = MSE(v_pred, x1 - x0). LoRA on q/k/v/o_proj (self_attn + cross_attn)
//   at every layer, rank/alpha configurable, AdamW with bias-corrected LR.
//
// Design: the frozen DiT backbone (src/dit.h) and the per-layer LoRA A/B +
// AdamW momentum tensors both live in persistent WeightCtx-backed backend
// buffers, allocated once. Each training step builds a *fresh* ggml_context
// (shapes vary with T) containing only that step's activations, referencing
// the persistent tensors by pointer. This sidesteps ggml-opt's high-level
// dataset/epoch API entirely: that API keys its persistent AdamW momenta by
// *node position* in a graph rebuilt once, which is unsafe here because our
// forward graph's node count/order shifts with T. Keying our own momentum
// tensors by parameter *identity* (one m/v pair per LoRA tensor, held in our
// own struct) avoids that footgun.

#include "dit-graph.h"
#include "ggml-alloc.h"
#include "safetensors-write.h"
#include "static-graph.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Turbo model shift=3.0 discrete timesteps (8 steps, same as inference).
// Matches TURBO_SHIFT3_TIMESTEPS in the Python trainer.
static const float DIT_TRAIN_TURBO_SHIFT3_TIMESTEPS[8] = {
    1.0f, 0.9545454545454546f, 0.9f, 0.8333333333333334f, 0.75f, 0.6428571428571429f, 0.5f, 0.3f,
};

struct DiTTrainConfig {
    int   rank          = 8;
    int   alpha         = 16;
    float lr            = 1e-4f;
    float beta1         = 0.9f;
    float beta2         = 0.999f;
    float eps           = 1e-8f;
    float weight_decay  = 0.01f;
};

struct DiTTrainProjMomentum {
    struct ggml_tensor * A_m = nullptr;
    struct ggml_tensor * A_v = nullptr;
    struct ggml_tensor * B_m = nullptr;
    struct ggml_tensor * B_v = nullptr;
};

struct DiTTrainLayerMomentum {
    DiTTrainProjMomentum sa_q, sa_k, sa_v, sa_o;
    DiTTrainProjMomentum ca_q, ca_k, ca_v, ca_o;
};

// Host-side gradient accumulator for one projection (A, B). Summed across
// `dit_train_forward_backward` calls between two `dit_train_optimizer_step`
// calls, then divided by the micro-batch count and applied. Plain host
// vectors, not backend tensors: each micro-batch's backward pass produces
// its gradient in a *fresh*, per-step ggml_context, so accumulating on the
// host (get, add, keep) is simpler than wiring a persistent add-in-graph
// across contexts, and the tensors involved are small (rank * hidden).
struct DiTTrainProjGradAccum {
    std::vector<float> A;
    std::vector<float> B;
};

struct DiTTrainLayerGradAccum {
    DiTTrainProjGradAccum sa_q, sa_k, sa_v, sa_o;
    DiTTrainProjGradAccum ca_q, ca_k, ca_v, ca_o;
};

struct DiTTrain {
    DiTGGML                            dit;    // frozen backbone, F32, no-FA (src/dit.h::dit_ggml_load_train)
    DiTTrainConfig                      cfg;
    std::vector<DiTLoraLayer>          lora;   // n_layers entries; A/B/scaling, fed straight to dit_ggml_build_graph
    std::vector<DiTTrainLayerMomentum> mom;    // n_layers entries; AdamW m/v, parallel to `lora`
    std::vector<DiTTrainLayerGradAccum> grad_accum;  // n_layers entries; host-side, parallel to `lora`
    int                                 accum_count = 0;  // micro-batches accumulated since the last optimizer step
    WeightCtx                           lora_wctx;  // persistent backend buffer for lora + mom + adamw_params
    struct ggml_tensor *                adamw_params = nullptr;  // [7] F32: alpha,beta1,beta2,eps,wd,beta1h,beta2h
    int64_t                             adam_iter    = 0;
};

// Fixed order used everywhere the 8 self_attn/cross_attn q/k/v/o_proj
// projections of one layer need to be walked as parallel arrays (already
// the pattern dit_train_init/dit_train_step use below).
#define DIT_TRAIN_PROJ_PTRS(layer) \
    { &(layer).sa_q, &(layer).sa_k, &(layer).sa_v, &(layer).sa_o, &(layer).ca_q, &(layer).ca_k, &(layer).ca_v, &(layer).ca_o }

// GGUF decoder tensor name fragment for projection j (0..7, same order as
// DIT_TRAIN_PROJ_PTRS). Matches src/adapter-merge.h's lora_base_name()
// mapping so checkpoints load unmodified through the existing inference path.
static const char * dit_train_proj_kind(int j) {
    static const char * kinds[8] = { "self_attn.q_proj",  "self_attn.k_proj",  "self_attn.v_proj",
                                     "self_attn.o_proj",  "cross_attn.q_proj", "cross_attn.k_proj",
                                     "cross_attn.v_proj", "cross_attn.o_proj" };
    return kinds[j];
}

// Allocate one LoRA projection (A, B) + its AdamW momentum (A_m, A_v, B_m,
// B_v) inside wctx. A ~ N(0, 1/sqrt(in_feat)) (breaks symmetry so B receives
// a nonzero gradient on step 1), B = 0 (standard LoRA init: the adapter
// starts as a no-op). Momentum starts at zero.
static void dit_train_alloc_proj(WeightCtx *          wctx,
                                 DiTLoraProj *        lp,
                                 DiTTrainProjMomentum * mp,
                                 int                   in_feat,
                                 int                   out_feat,
                                 int                   rank,
                                 float                 scaling,
                                 std::mt19937 &        rng) {
    lp->scaling = scaling;
    lp->A       = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, in_feat, rank);
    lp->B       = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, rank, out_feat);
    ggml_set_param(lp->A);
    ggml_set_param(lp->B);

    mp->A_m = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, in_feat, rank);
    mp->A_v = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, in_feat, rank);
    mp->B_m = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, rank, out_feat);
    mp->B_v = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, rank, out_feat);

    size_t a_n = (size_t) in_feat * rank;
    size_t b_n = (size_t) out_feat * rank;

    auto                             a_buf = std::make_unique<float[]>(a_n);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float) in_feat));
    for (size_t i = 0; i < a_n; i++) {
        a_buf[i] = dist(rng);
    }
    wctx->pending.push_back({ lp->A, a_buf.get(), a_n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(a_buf));

    auto b_buf = std::make_unique<float[]>(b_n);
    std::fill(b_buf.get(), b_buf.get() + b_n, 0.0f);
    wctx->pending.push_back({ lp->B, b_buf.get(), b_n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(b_buf));

    auto zero_stage = [&](struct ggml_tensor * t, size_t n) {
        auto buf = std::make_unique<float[]>(n);
        std::fill(buf.get(), buf.get() + n, 0.0f);
        wctx->pending.push_back({ t, buf.get(), n * sizeof(float), 0 });
        wctx->staging.push_back(std::move(buf));
    };
    zero_stage(mp->A_m, a_n);
    zero_stage(mp->A_v, a_n);
    zero_stage(mp->B_m, b_n);
    zero_stage(mp->B_v, b_n);
}

// Load the frozen DiT (F32, train mode) and allocate LoRA + AdamW state for
// every self_attn/cross_attn q/k/v/o_proj at every layer.
// max_layers: 0 for the full model, a positive cap for cheap structural tests.
static bool dit_train_init(DiTTrain *             t,
                           const char *           gguf_path,
                           const DiTTrainConfig & cfg,
                           int                    max_layers = 0,
                           uint32_t               seed       = 42) {
    t->cfg = cfg;
    if (!dit_ggml_load_train(&t->dit, gguf_path, max_layers)) {
        return false;
    }

    int n_layers = t->dit.cfg.n_layers;
    t->lora.resize(n_layers);
    t->mom.resize(n_layers);
    t->grad_accum.resize(n_layers);

    // 8 projections/layer * (A, B, A_m, A_v, B_m, B_v) + adamw_params
    int n_tensors = n_layers * 8 * 6 + 1;
    wctx_init(&t->lora_wctx, n_tensors);

    std::mt19937 rng(seed);
    float        scaling = (float) cfg.alpha / (float) cfg.rank;

    auto in_of  = [](struct ggml_tensor * w) { return (int) w->ne[0]; };
    auto out_of = [](struct ggml_tensor * w) { return (int) w->ne[1]; };

    for (int i = 0; i < n_layers; i++) {
        DiTGGMLLayer &          ly = t->dit.layers[i];
        DiTLoraLayer &          lp = t->lora[i];
        DiTTrainLayerMomentum & mp = t->mom[i];

        dit_train_alloc_proj(&t->lora_wctx, &lp.sa_q, &mp.sa_q, in_of(ly.sa_q_proj), out_of(ly.sa_q_proj), cfg.rank,
                             scaling, rng);
        dit_train_alloc_proj(&t->lora_wctx, &lp.sa_k, &mp.sa_k, in_of(ly.sa_k_proj), out_of(ly.sa_k_proj), cfg.rank,
                             scaling, rng);
        dit_train_alloc_proj(&t->lora_wctx, &lp.sa_v, &mp.sa_v, in_of(ly.sa_v_proj), out_of(ly.sa_v_proj), cfg.rank,
                             scaling, rng);
        dit_train_alloc_proj(&t->lora_wctx, &lp.sa_o, &mp.sa_o, in_of(ly.sa_o_proj), out_of(ly.sa_o_proj), cfg.rank,
                             scaling, rng);
        dit_train_alloc_proj(&t->lora_wctx, &lp.ca_q, &mp.ca_q, in_of(ly.ca_q_proj), out_of(ly.ca_q_proj), cfg.rank,
                             scaling, rng);
        dit_train_alloc_proj(&t->lora_wctx, &lp.ca_k, &mp.ca_k, in_of(ly.ca_k_proj), out_of(ly.ca_k_proj), cfg.rank,
                             scaling, rng);
        dit_train_alloc_proj(&t->lora_wctx, &lp.ca_v, &mp.ca_v, in_of(ly.ca_v_proj), out_of(ly.ca_v_proj), cfg.rank,
                             scaling, rng);
        dit_train_alloc_proj(&t->lora_wctx, &lp.ca_o, &mp.ca_o, in_of(ly.ca_o_proj), out_of(ly.ca_o_proj), cfg.rank,
                             scaling, rng);

        DiTLoraProj * projs[8] = DIT_TRAIN_PROJ_PTRS(lp);
        DiTTrainProjGradAccum * gprojs[8] = DIT_TRAIN_PROJ_PTRS(t->grad_accum[i]);
        for (int j = 0; j < 8; j++) {
            gprojs[j]->A.assign(ggml_nelements(projs[j]->A), 0.0f);
            gprojs[j]->B.assign(ggml_nelements(projs[j]->B), 0.0f);
        }
    }

    t->adamw_params = ggml_new_tensor_1d(t->lora_wctx.ctx, GGML_TYPE_F32, 7);
    ggml_set_name(t->adamw_params, "adamw_params");
    auto adamw_init = std::make_unique<float[]>(7);
    std::fill(adamw_init.get(), adamw_init.get() + 7, 0.0f);
    t->lora_wctx.pending.push_back({ t->adamw_params, adamw_init.get(), 7 * sizeof(float), 0 });
    t->lora_wctx.staging.push_back(std::move(adamw_init));

    if (!wctx_alloc(&t->lora_wctx, t->dit.backend)) {
        return false;
    }

    int64_t total_params = 0;
    for (int i = 0; i < n_layers; i++) {
        DiTLoraProj * projs[8] = DIT_TRAIN_PROJ_PTRS(t->lora[i]);
        for (DiTLoraProj * p : projs) {
            total_params += ggml_nelements(p->A) + ggml_nelements(p->B);
        }
    }
    fprintf(stderr, "[Train] LoRA: rank=%d alpha=%d, %lld trainable params across %d layers\n", cfg.rank, cfg.alpha,
            (long long) total_params, n_layers);
    return true;
}

static void dit_train_free(DiTTrain * t) {
    wctx_free(&t->lora_wctx);
    dit_ggml_free(&t->dit);
    *t = {};
}

// Forward (DiT + LoRA), flow-matching MSE loss, backward. Accumulates every
// LoRA tensor's gradient into t->grad_accum (host-side sum) and bumps
// t->accum_count; does NOT touch the parameters or momenta. Call
// dit_train_optimizer_step() after `grad_accum` calls to actually apply the
// averaged gradient. Builds a fresh graph every call (T/enc_S may differ
// per sample); the frozen backbone and all LoRA/momentum tensors are
// persistent and referenced by pointer, not rebuilt.
//
// target_latents:  [T, 64]      x0, VAE-encoded audio (row major, ggml ne=(64,T))
// context_latents: [T, 128]     src||mask context (silence+ones(1.0) for text2music)
// enc_hidden_data: [H_enc, enc_S]  condition encoder output (ggml ne=(H_enc,enc_S))
// noise:           [T, 64]      x1, sampled once per step by the caller
// t_val:           flow matching timestep (shared t=t_r, turbo convention)
static float dit_train_forward_backward(DiTTrain *   t,
                                        int          T,
                                        int          enc_S,
                                        const float * target_latents,
                                        const float * context_latents,
                                        const float * enc_hidden_data,
                                        const float * noise,
                                        float         t_val) {
    DiTGGMLConfig & c      = t->dit.cfg;
    int             Oc     = c.out_channels;         // 64
    int             ctx_ch = c.in_channels - Oc;     // 128
    int             in_ch  = c.in_channels;          // 192
    int             S      = T / c.patch_size;
    int             H_enc  = (int) t->dit.cond_emb_w->ne[0];
    int             n_layers = c.n_layers;

    // Fresh per-step context: forward + backward + n_layers*8*2 opt-step nodes.
    size_t                ctx_size = ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(32768, true) +
                          (size_t) 4 * 1024 * 1024;
    std::vector<uint8_t>   ctx_buf(ctx_size);
    struct ggml_init_params gparams = { ctx_size, ctx_buf.data(), true };
    struct ggml_context *   ctx     = ggml_init(gparams);
    if (!ctx) {
        fprintf(stderr, "[Train] FATAL: ggml_init failed for step context (%zu bytes)\n", ctx_size);
        exit(1);
    }

    struct ggml_tensor * t_input  = nullptr;
    struct ggml_tensor * t_output = nullptr;
    struct ggml_cgraph *  gf =
        dit_ggml_build_graph(&t->dit, ctx, T, enc_S, /*N=*/1, &t_input, &t_output, t->lora.data(), /*want_grads=*/true);

    // Loss: MSE(v_pred, x1 - x0), mean over every element (matches the
    // Python trainer's F.mse_loss default reduction).
    struct ggml_tensor * flow_target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Oc, T, 1);
    ggml_set_name(flow_target, "flow_target");
    ggml_set_input(flow_target);

    struct ggml_tensor * diff = ggml_sub(ctx, t_output, flow_target);
    struct ggml_tensor * sq   = ggml_sqr(ctx, diff);
    struct ggml_tensor * loss = ggml_sum(ctx, sq);
    loss                      = ggml_scale(ctx, loss, 1.0f / (float) ggml_nelements(t_output));
    ggml_set_name(loss, "loss");
    ggml_set_loss(loss);
    ggml_set_output(loss);
    ggml_build_forward_expand(gf, loss);

    ggml_build_backward_expand(ctx, gf, nullptr);

    // Locate every LoRA tensor's gradient now (graph node, valid once
    // compute runs below); read back and accumulate after compute.
    struct GradHandle {
        struct ggml_tensor *    gA;
        struct ggml_tensor *    gB;
        DiTTrainProjGradAccum * accum;
    };
    std::vector<GradHandle> grad_handles;
    grad_handles.reserve((size_t) n_layers * 8);
    for (int i = 0; i < n_layers; i++) {
        DiTLoraProj *           projs[8]  = DIT_TRAIN_PROJ_PTRS(t->lora[i]);
        DiTTrainProjGradAccum * gprojs[8] = DIT_TRAIN_PROJ_PTRS(t->grad_accum[i]);
        for (int j = 0; j < 8; j++) {
            struct ggml_tensor * gA = ggml_graph_get_grad(gf, projs[j]->A);
            struct ggml_tensor * gB = ggml_graph_get_grad(gf, projs[j]->B);
            if (!gA || !gB) {
                fprintf(stderr, "[Train] FATAL: no gradient for layer %d proj %d (A=%p B=%p)\n", i, j, (void *) gA,
                        (void *) gB);
                exit(1);
            }
            grad_handles.push_back({ gA, gB, gprojs[j] });
        }
    }

    // ---- Inputs -------------------------------------------------------
    struct ggml_tensor * t_enc      = ggml_graph_get_tensor(gf, "enc_hidden");
    struct ggml_tensor * t_t        = ggml_graph_get_tensor(gf, "t");
    struct ggml_tensor * t_tr       = ggml_graph_get_tensor(gf, "t_r");
    struct ggml_tensor * t_pos      = ggml_graph_get_tensor(gf, "positions");
    struct ggml_tensor * t_sa_mask  = ggml_graph_get_tensor(gf, "sa_mask_sw");
    struct ggml_tensor * t_ca_mask  = ggml_graph_get_tensor(gf, "ca_mask");

    StaticGraph sg;
    if (!static_graph_alloc(&sg, t->dit.backend, t->dit.sched, gf)) {
        fprintf(stderr, "[Train] FATAL: failed to allocate training graph\n");
        exit(1);
    }

    // Seed dL/dL = 1: a fresh backward graph's loss gradient accumulator
    // is NOT seeded automatically (that's normally ggml_graph_reset()'s
    // job). We don't call the blanket reset here because it would also
    // zero every GGML_OP_OPT_STEP_ADAMW node's momenta in
    // dit_train_optimizer_step()'s graph -- not relevant to *this*
    // function (which builds no such nodes), but the two share the
    // pattern so it's noted once, here, for both.
    struct ggml_tensor * loss_grad_acc = ggml_graph_get_grad_acc(gf, loss);
    if (!loss_grad_acc) {
        fprintf(stderr, "[Train] FATAL: loss has no gradient accumulator\n");
        exit(1);
    }
    {
        float onef = 1.0f;
        ggml_backend_tensor_set(loss_grad_acc, &onef, 0, sizeof(float));
    }

    // input_latents = concat(context[128], xt[64]) per frame; xt = t*x1 + (1-t)*x0
    std::vector<float> input_buf((size_t) in_ch * T);
    std::vector<float> flow_buf((size_t) Oc * T);
    for (int ti = 0; ti < T; ti++) {
        memcpy(&input_buf[(size_t) ti * in_ch], &context_latents[(size_t) ti * ctx_ch], ctx_ch * sizeof(float));
        for (int ci = 0; ci < Oc; ci++) {
            float x0                                              = target_latents[(size_t) ti * Oc + ci];
            float x1                                              = noise[(size_t) ti * Oc + ci];
            float xt                                              = t_val * x1 + (1.0f - t_val) * x0;
            input_buf[(size_t) ti * in_ch + ctx_ch + ci]          = xt;
            flow_buf[(size_t) ti * Oc + ci]                       = x1 - x0;
        }
    }
    ggml_backend_tensor_set(t_input, input_buf.data(), 0, input_buf.size() * sizeof(float));
    ggml_backend_tensor_set(flow_target, flow_buf.data(), 0, flow_buf.size() * sizeof(float));
    ggml_backend_tensor_set(t_enc, enc_hidden_data, 0, (size_t) H_enc * enc_S * sizeof(float));
    ggml_backend_tensor_set(t_t, &t_val, 0, sizeof(float));
    ggml_backend_tensor_set(t_tr, &t_val, 0, sizeof(float));

    std::vector<int32_t> pos_data((size_t) S);
    for (int i = 0; i < S; i++) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(t_pos, pos_data.data(), 0, (size_t) S * sizeof(int32_t));

    // No padding in this synthetic sample: masks are all-valid. Sliding
    // window still applies within S for layer_type=0 layers (matches inference).
    int                   win = c.sliding_window;
    std::vector<uint16_t> sa_data((size_t) S * S);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < S; ki++) {
            int  dist   = (qi > ki) ? (qi - ki) : (ki - qi);
            bool in_win = (win <= 0) || (S <= win) || (dist <= win);
            sa_data[(size_t) qi * S + ki] = ggml_fp32_to_fp16(in_win ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(t_sa_mask, sa_data.data(), 0, sa_data.size() * sizeof(uint16_t));

    std::vector<uint16_t> ca_data((size_t) enc_S * S, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, ca_data.size() * sizeof(uint16_t));

    enum ggml_status st = static_graph_compute(&sg, t->dit.backend, t->dit.sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[Train] FATAL: graph compute failed (status=%d)\n", (int) st);
        exit(1);
    }

    float loss_val = 0.0f;
    ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));

    // Accumulate every LoRA tensor's freshly computed gradient on the host.
    std::vector<float> tmp;
    for (const GradHandle & h : grad_handles) {
        tmp.resize(h.accum->A.size());
        ggml_backend_tensor_get(h.gA, tmp.data(), 0, tmp.size() * sizeof(float));
        for (size_t k = 0; k < tmp.size(); k++) {
            h.accum->A[k] += tmp[k];
        }
        tmp.resize(h.accum->B.size());
        ggml_backend_tensor_get(h.gB, tmp.data(), 0, tmp.size() * sizeof(float));
        for (size_t k = 0; k < tmp.size(); k++) {
            h.accum->B[k] += tmp[k];
        }
    }
    t->accum_count++;

    static_graph_release(&sg, t->dit.sched);
    ggml_free(ctx);
    return loss_val;
}

// Forward-only pass: DiT + LoRA (current trained values) + flow-matching
// MSE loss, no backward, no parameter changes. Used for validation.
static float dit_train_eval(DiTTrain *   t,
                            int          T,
                            int          enc_S,
                            const float * target_latents,
                            const float * context_latents,
                            const float * enc_hidden_data,
                            const float * noise,
                            float         t_val) {
    DiTGGMLConfig & c      = t->dit.cfg;
    int             Oc     = c.out_channels;
    int             ctx_ch = c.in_channels - Oc;
    int             in_ch  = c.in_channels;
    int             S      = T / c.patch_size;
    int             H_enc  = (int) t->dit.cond_emb_w->ne[0];

    size_t                  ctx_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(8192, false) +
                          (size_t) 2 * 1024 * 1024;
    std::vector<uint8_t>    ctx_buf(ctx_size);
    struct ggml_init_params gparams = { ctx_size, ctx_buf.data(), true };
    struct ggml_context *   ctx     = ggml_init(gparams);
    if (!ctx) {
        fprintf(stderr, "[Train] FATAL: ggml_init failed for eval context (%zu bytes)\n", ctx_size);
        exit(1);
    }

    struct ggml_tensor * t_input  = nullptr;
    struct ggml_tensor * t_output = nullptr;
    struct ggml_cgraph *  gf =
        dit_ggml_build_graph(&t->dit, ctx, T, enc_S, /*N=*/1, &t_input, &t_output, t->lora.data(), /*want_grads=*/false);

    struct ggml_tensor * flow_target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Oc, T, 1);
    ggml_set_name(flow_target, "flow_target");
    ggml_set_input(flow_target);

    struct ggml_tensor * diff = ggml_sub(ctx, t_output, flow_target);
    struct ggml_tensor * sq   = ggml_sqr(ctx, diff);
    struct ggml_tensor * loss = ggml_sum(ctx, sq);
    loss                      = ggml_scale(ctx, loss, 1.0f / (float) ggml_nelements(t_output));
    ggml_set_name(loss, "loss");
    ggml_set_output(loss);
    ggml_build_forward_expand(gf, loss);

    struct ggml_tensor * t_enc     = ggml_graph_get_tensor(gf, "enc_hidden");
    struct ggml_tensor * t_t       = ggml_graph_get_tensor(gf, "t");
    struct ggml_tensor * t_tr      = ggml_graph_get_tensor(gf, "t_r");
    struct ggml_tensor * t_pos     = ggml_graph_get_tensor(gf, "positions");
    struct ggml_tensor * t_sa_mask = ggml_graph_get_tensor(gf, "sa_mask_sw");
    struct ggml_tensor * t_ca_mask = ggml_graph_get_tensor(gf, "ca_mask");

    StaticGraph sg;
    if (!static_graph_alloc(&sg, t->dit.backend, t->dit.sched, gf)) {
        fprintf(stderr, "[Train] FATAL: failed to allocate eval graph\n");
        exit(1);
    }

    std::vector<float> input_buf((size_t) in_ch * T);
    std::vector<float> flow_buf((size_t) Oc * T);
    for (int ti = 0; ti < T; ti++) {
        memcpy(&input_buf[(size_t) ti * in_ch], &context_latents[(size_t) ti * ctx_ch], ctx_ch * sizeof(float));
        for (int ci = 0; ci < Oc; ci++) {
            float x0                                     = target_latents[(size_t) ti * Oc + ci];
            float x1                                     = noise[(size_t) ti * Oc + ci];
            float xt                                     = t_val * x1 + (1.0f - t_val) * x0;
            input_buf[(size_t) ti * in_ch + ctx_ch + ci] = xt;
            flow_buf[(size_t) ti * Oc + ci]               = x1 - x0;
        }
    }
    ggml_backend_tensor_set(t_input, input_buf.data(), 0, input_buf.size() * sizeof(float));
    ggml_backend_tensor_set(flow_target, flow_buf.data(), 0, flow_buf.size() * sizeof(float));
    ggml_backend_tensor_set(t_enc, enc_hidden_data, 0, (size_t) H_enc * enc_S * sizeof(float));
    ggml_backend_tensor_set(t_t, &t_val, 0, sizeof(float));
    ggml_backend_tensor_set(t_tr, &t_val, 0, sizeof(float));

    std::vector<int32_t> pos_data((size_t) S);
    for (int i = 0; i < S; i++) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(t_pos, pos_data.data(), 0, (size_t) S * sizeof(int32_t));

    int                   win = c.sliding_window;
    std::vector<uint16_t> sa_data((size_t) S * S);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < S; ki++) {
            int  dist   = (qi > ki) ? (qi - ki) : (ki - qi);
            bool in_win = (win <= 0) || (S <= win) || (dist <= win);
            sa_data[(size_t) qi * S + ki] = ggml_fp32_to_fp16(in_win ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(t_sa_mask, sa_data.data(), 0, sa_data.size() * sizeof(uint16_t));

    std::vector<uint16_t> ca_data((size_t) enc_S * S, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, ca_data.size() * sizeof(uint16_t));

    enum ggml_status st = static_graph_compute(&sg, t->dit.backend, t->dit.sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[Train] FATAL: eval graph compute failed (status=%d)\n", (int) st);
        exit(1);
    }

    float loss_val = 0.0f;
    ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));

    static_graph_release(&sg, t->dit.sched);
    ggml_free(ctx);
    return loss_val;
}

// Apply the averaged accumulated gradient (t->accum_count micro-batches
// worth) via AdamW to every LoRA tensor, then reset the accumulators and
// bump the AdamW step counter. Builds its own small graph (no forward pass,
// just one GGML_OP_OPT_STEP_ADAMW view-node per tensor): cheap regardless
// of how many dit_train_forward_backward calls preceded it.
static void dit_train_optimizer_step(DiTTrain * t) {
    if (t->accum_count == 0) {
        fprintf(stderr, "[Train] WARNING: dit_train_optimizer_step called with no accumulated gradient, skipping\n");
        return;
    }
    int n_layers = t->dit.cfg.n_layers;
    float inv_count = 1.0f / (float) t->accum_count;

    t->adam_iter++;
    float beta1h        = 1.0f / (1.0f - powf(t->cfg.beta1, (float) t->adam_iter));
    float beta2h        = 1.0f / (1.0f - powf(t->cfg.beta2, (float) t->adam_iter));
    float adamw_host[7] = { t->cfg.lr, t->cfg.beta1, t->cfg.beta2, t->cfg.eps, t->cfg.weight_decay, beta1h, beta2h };
    ggml_backend_tensor_set(t->adamw_params, adamw_host, 0, 7 * sizeof(float));

    // Graph "size" caps both nodes[] AND leafs[] at the same capacity. Nodes:
    // 2 opt-step ops per proj (stepA, stepB). Leaves: 8 per proj (A, B,
    // gA_in, gB_in, A_m, A_v, B_m, B_v) plus the one shared adamw_params --
    // leaves dominate, size for those with margin.
    size_t graph_size = (size_t) (n_layers * 8 * 8 + 32);
    size_t ctx_size    = ggml_tensor_overhead() * (size_t) (n_layers * 8 * 4 + 16) +
                       ggml_graph_overhead_custom(graph_size, false) + (size_t) 256 * 1024;
    std::vector<uint8_t>    ctx_buf(ctx_size);
    struct ggml_init_params gparams = { ctx_size, ctx_buf.data(), true };
    struct ggml_context *   ctx     = ggml_init(gparams);
    if (!ctx) {
        fprintf(stderr, "[Train] FATAL: ggml_init failed for optimizer-step context (%zu bytes)\n", ctx_size);
        exit(1);
    }
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, /*grads=*/false);

    struct GradUpload {
        struct ggml_tensor * grad_in;
        std::vector<float> * host;
    };
    std::vector<GradUpload> uploads;
    uploads.reserve((size_t) n_layers * 16);

    for (int i = 0; i < n_layers; i++) {
        DiTLoraProj *           projs[8]  = DIT_TRAIN_PROJ_PTRS(t->lora[i]);
        DiTTrainProjMomentum *  moms[8]   = DIT_TRAIN_PROJ_PTRS(t->mom[i]);
        DiTTrainProjGradAccum * gprojs[8] = DIT_TRAIN_PROJ_PTRS(t->grad_accum[i]);

        for (int j = 0; j < 8; j++) {
            DiTLoraProj * p = projs[j];

            struct ggml_tensor * gA_in = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(p->A), p->A->ne);
            struct ggml_tensor * gB_in = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(p->B), p->B->ne);
            ggml_set_input(gA_in);
            ggml_set_input(gB_in);
            uploads.push_back({ gA_in, &gprojs[j]->A });
            uploads.push_back({ gB_in, &gprojs[j]->B });

            struct ggml_tensor * stepA =
                ggml_opt_step_adamw(ctx, p->A, gA_in, moms[j]->A_m, moms[j]->A_v, t->adamw_params);
            struct ggml_tensor * stepB =
                ggml_opt_step_adamw(ctx, p->B, gB_in, moms[j]->B_m, moms[j]->B_v, t->adamw_params);
            ggml_build_forward_expand(gf, stepA);
            ggml_build_forward_expand(gf, stepB);
        }
    }

    StaticGraph sg;
    if (!static_graph_alloc(&sg, t->dit.backend, t->dit.sched, gf)) {
        fprintf(stderr, "[Train] FATAL: failed to allocate optimizer-step graph\n");
        exit(1);
    }

    for (const GradUpload & u : uploads) {
        for (float & v : *u.host) {
            v *= inv_count;
        }
        ggml_backend_tensor_set(u.grad_in, u.host->data(), 0, u.host->size() * sizeof(float));
    }

    enum ggml_status st = static_graph_compute(&sg, t->dit.backend, t->dit.sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[Train] FATAL: optimizer-step graph compute failed (status=%d)\n", (int) st);
        exit(1);
    }

    static_graph_release(&sg, t->dit.sched);
    ggml_free(ctx);

    for (int i = 0; i < n_layers; i++) {
        DiTTrainProjGradAccum * gprojs[8] = DIT_TRAIN_PROJ_PTRS(t->grad_accum[i]);
        for (int j = 0; j < 8; j++) {
            std::fill(gprojs[j]->A.begin(), gprojs[j]->A.end(), 0.0f);
            std::fill(gprojs[j]->B.begin(), gprojs[j]->B.end(), 0.0f);
        }
    }
    t->accum_count = 0;
}

// Save the current LoRA weights as a PEFT adapter directory: a drop-in
// build for src/adapter-merge.h (verified against the real merge path in
// tests/test-lora-roundtrip.cpp). dir is created if missing.
static bool dit_train_save_checkpoint(DiTTrain * t, const std::string & dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        fprintf(stderr, "[Train] FATAL: could not create %s: %s\n", dir.c_str(), ec.message().c_str());
        return false;
    }

    int n_layers = t->dit.cfg.n_layers;

    std::vector<STWriteEntry>       entries;
    std::vector<std::vector<float>> bufs;  // keep data alive until st_write
    entries.reserve((size_t) n_layers * 16);
    bufs.reserve((size_t) n_layers * 16);

    for (int i = 0; i < n_layers; i++) {
        DiTLoraProj * projs[8] = DIT_TRAIN_PROJ_PTRS(t->lora[i]);
        for (int j = 0; j < 8; j++) {
            DiTLoraProj * p = projs[j];
            int64_t       rank     = p->A->ne[1];
            int64_t       in_feat  = p->A->ne[0];
            int64_t       out_feat = p->B->ne[1];

            std::string base = "base_model.model.layers." + std::to_string(i) + "." + dit_train_proj_kind(j);

            bufs.emplace_back((size_t) ggml_nelements(p->A));
            ggml_backend_tensor_get(p->A, bufs.back().data(), 0, ggml_nbytes(p->A));
            entries.push_back(
                { base + ".lora_A.weight", "F32", { rank, in_feat }, bufs.back().data(), ggml_nbytes(p->A) });

            bufs.emplace_back((size_t) ggml_nelements(p->B));
            ggml_backend_tensor_get(p->B, bufs.back().data(), 0, ggml_nbytes(p->B));
            entries.push_back(
                { base + ".lora_B.weight", "F32", { out_feat, rank }, bufs.back().data(), ggml_nbytes(p->B) });
        }
    }

    std::string st_path = dir + "/adapter_model.safetensors";
    if (!st_write(st_path, entries)) {
        return false;
    }

    std::string cfg_path = dir + "/adapter_config.json";
    FILE *      cfg_f    = fopen(cfg_path.c_str(), "wb");
    if (!cfg_f) {
        fprintf(stderr, "[Train] FATAL: could not write %s\n", cfg_path.c_str());
        return false;
    }
    fprintf(cfg_f,
            "{\"r\":%d,\"lora_alpha\":%d,\"lora_dropout\":0.0,"
            "\"target_modules\":[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"],\"bias\":\"none\"}\n",
            t->cfg.rank, t->cfg.alpha);
    fclose(cfg_f);

    fprintf(stderr, "[Train] Checkpoint saved: %s (%d layers, rank=%d, alpha=%d)\n", dir.c_str(), n_layers,
            t->cfg.rank, t->cfg.alpha);
    return true;
}
