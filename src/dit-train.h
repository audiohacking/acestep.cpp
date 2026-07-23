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
#include "static-graph.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
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

struct DiTTrain {
    DiTGGML                            dit;    // frozen backbone, F32, no-FA (src/dit.h::dit_ggml_load_train)
    DiTTrainConfig                      cfg;
    std::vector<DiTLoraLayer>          lora;   // n_layers entries; A/B/scaling, fed straight to dit_ggml_build_graph
    std::vector<DiTTrainLayerMomentum> mom;    // n_layers entries; AdamW m/v, parallel to `lora`
    WeightCtx                           lora_wctx;  // persistent backend buffer for lora + mom + adamw_params
    struct ggml_tensor *                adamw_params = nullptr;  // [7] F32: alpha,beta1,beta2,eps,wd,beta1h,beta2h
    int64_t                             adam_iter    = 0;
};

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
        DiTLoraProj * projs[8] = { &t->lora[i].sa_q, &t->lora[i].sa_k, &t->lora[i].sa_v, &t->lora[i].sa_o,
                                  &t->lora[i].ca_q, &t->lora[i].ca_k, &t->lora[i].ca_v, &t->lora[i].ca_o };
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

// One training step: forward (DiT + LoRA), flow-matching MSE loss, backward,
// AdamW update on every LoRA A/B tensor. Builds a fresh graph every call
// (T/enc_S may differ per sample); the frozen backbone and all LoRA/momentum
// tensors are persistent and referenced by pointer, not rebuilt.
//
// target_latents:  [T, 64]      x0, VAE-encoded audio (row major, ggml ne=(64,T))
// context_latents: [T, 128]     src||mask context (silence+ones(1.0) for text2music)
// enc_hidden_data: [H_enc, enc_S]  condition encoder output (ggml ne=(H_enc,enc_S))
// noise:           [T, 64]      x1, sampled once per step by the caller
// t_val:           flow matching timestep (shared t=t_r, turbo convention)
static float dit_train_step(DiTTrain *   t,
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

    // AdamW step per LoRA tensor. Params tracked by identity (this struct's
    // own pointers), not by graph node position: see file header.
    t->adam_iter++;
    float beta1h = 1.0f / (1.0f - powf(t->cfg.beta1, (float) t->adam_iter));
    float beta2h = 1.0f / (1.0f - powf(t->cfg.beta2, (float) t->adam_iter));
    float adamw_host[7] = { t->cfg.lr, t->cfg.beta1, t->cfg.beta2, t->cfg.eps, t->cfg.weight_decay, beta1h, beta2h };

    for (int i = 0; i < n_layers; i++) {
        DiTLoraLayer &          lp   = t->lora[i];
        DiTTrainLayerMomentum & mp   = t->mom[i];
        DiTLoraProj *           projs[8] = { &lp.sa_q, &lp.sa_k, &lp.sa_v, &lp.sa_o, &lp.ca_q, &lp.ca_k, &lp.ca_v, &lp.ca_o };
        DiTTrainProjMomentum *  moms[8]  = { &mp.sa_q, &mp.sa_k, &mp.sa_v, &mp.sa_o, &mp.ca_q, &mp.ca_k, &mp.ca_v, &mp.ca_o };

        for (int j = 0; j < 8; j++) {
            DiTLoraProj *          p = projs[j];
            DiTTrainProjMomentum * m = moms[j];

            struct ggml_tensor * gA = ggml_graph_get_grad(gf, p->A);
            struct ggml_tensor * gB = ggml_graph_get_grad(gf, p->B);
            if (!gA || !gB) {
                fprintf(stderr, "[Train] FATAL: no gradient for layer %d proj %d (A=%p B=%p)\n", i, j, (void *) gA,
                        (void *) gB);
                exit(1);
            }
            struct ggml_tensor * stepA = ggml_opt_step_adamw(ctx, p->A, gA, m->A_m, m->A_v, t->adamw_params);
            struct ggml_tensor * stepB = ggml_opt_step_adamw(ctx, p->B, gB, m->B_m, m->B_v, t->adamw_params);
            ggml_build_forward_expand(gf, stepA);
            ggml_build_forward_expand(gf, stepB);
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

    // Seed dL/dL = 1 (ggml_graph_reset() would do this too, but it also
    // zeroes every GGML_OP_OPT_STEP_ADAMW node's momenta (src[2]/src[3]) --
    // which here alias our *persistent* m/v tensors, so a blanket reset
    // would erase Adam's cross-step history every single call. Seed only
    // the loss accumulator directly instead.
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

    ggml_backend_tensor_set(t->adamw_params, adamw_host, 0, 7 * sizeof(float));

    enum ggml_status st = static_graph_compute(&sg, t->dit.backend, t->dit.sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[Train] FATAL: graph compute failed (status=%d)\n", (int) st);
        exit(1);
    }

    float loss_val = 0.0f;
    ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));

    static_graph_release(&sg, t->dit.sched);
    ggml_free(ctx);
    return loss_val;
}
