// test-grad-compare.cpp: dump a single training step's LoRA gradient (real
// preprocessed smp, fixed LoRA A init + noise + t, exported to files) so
// it can be cross-checked against Python's FixedLoRAModule.training_step()
// computing the identical step. Diagnostic tool, not a pass/fail test --
// see tmp/grad-compare/compare.py for the Python side.
//
// Usage:
//   ./test-grad-compare --models <dir> --smp <gguf> --dump <dir>
//                        [--rank N] [--alpha N] [--t F]

#include "dit-train-data.h"
#include "dit-train.h"
#include "model-registry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

static void write_raw(const std::string & path, const std::vector<float> & v) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[GradCompare] FATAL: could not write %s\n", path.c_str());
        exit(1);
    }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

// Mirrors dit_train_forward_backward (src/dit-train.h) but additionally
// dumps d(loss)/d(hidden_after_layerN) for every layer boundary -- localizes
// exactly where a backward-pass divergence against Python starts, instead of
// only seeing the final LoRA A/B gradient after everything (right or wrong)
// has flowed through the whole 24-layer stack.
static float dit_train_forward_backward_layer_grads(DiTTrain *          t,
                                                     int                  T,
                                                     int                  enc_S,
                                                     const float *        target_latents,
                                                     const float *        context_latents,
                                                     const float *        enc_hidden_data,
                                                     const float *        noise,
                                                     float                t_val,
                                                     const std::string & dump_dir) {
    DiTGGMLConfig & c        = t->dit.cfg;
    int             Oc       = c.out_channels;
    int             ctx_ch   = c.in_channels - Oc;
    int             in_ch    = c.in_channels;
    int             S        = T / c.patch_size;
    int             H_enc    = (int) t->dit.cond_emb_w->ne[0];
    int             n_layers = c.n_layers;

    size_t                ctx_size = ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(32768, true) +
                          (size_t) 4 * 1024 * 1024;
    std::vector<uint8_t>   ctx_buf(ctx_size);
    struct ggml_init_params gparams = { ctx_size, ctx_buf.data(), true };
    struct ggml_context *   ctx     = ggml_init(gparams);

    struct ggml_tensor * t_input  = nullptr;
    struct ggml_tensor * t_output = nullptr;
    g_dit_debug_layer_dumps = true;
    struct ggml_cgraph *  gf =
        dit_ggml_build_graph(&t->dit, ctx, T, enc_S, /*N=*/1, &t_input, &t_output, t->lora.data(), /*want_grads=*/true);
    g_dit_debug_layer_dumps = false;

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

    // Locate each layer's forward hidden tensor AND its gradient, marking
    // both as protected outputs *before* ggml_gallocr_alloc_graph runs.
    // Without ggml_set_output on the gradient tensor specifically, the
    // allocator is free to recycle its buffer for a later backward step
    // once it's "consumed" -- reading all 24 out after one single compute
    // call then returns 24 copies of whatever happened to be left in the
    // last-reused buffer (this was silently wrong before this fix: every
    // layer reported the exact same gradient magnitude, which is what
    // exposed the bug).
    std::vector<struct ggml_tensor *> hidden_tensors(n_layers, nullptr);
    std::vector<struct ggml_tensor *> hidden_grads(n_layers, nullptr);
    for (int i = 0; i < n_layers; i++) {
        char name[64];
        snprintf(name, sizeof(name), "hidden_after_layer%d", i);
        hidden_tensors[i] = ggml_graph_get_tensor(gf, name);
        if (hidden_tensors[i]) {
            hidden_grads[i] = ggml_graph_get_grad(gf, hidden_tensors[i]);
            if (hidden_grads[i]) {
                ggml_set_output(hidden_grads[i]);
            }
        }
    }

    // Sub-component debug tensors for layers 0-3 (dit-graph.h's dbg_out):
    // sa_input, sa_output, after_self_attn, ca_output, after_cross_attn,
    // ffn_output. Same buffer-protection requirement as hidden_tensors above.
    static const char * kSubNames[] = { "sa_input", "sa_output", "after_self_attn",
                                        "norm_ca", "ca_output", "after_cross_attn", "ffn_output",
                                        "ca_q_proj_out", "ca_k_proj_out", "ca_v_proj_out",
                                        "ca_query_states", "ca_key_states", "ca_value_states",
                                        "ca_attn_out_raw", "ca_attn_out_reshaped" };
    std::vector<struct ggml_tensor *> sub_tensors;
    std::vector<struct ggml_tensor *> sub_grads;
    std::vector<std::string>          sub_labels;
    for (int i = 0; i <= 3 && i < n_layers; i++) {
        for (const char * suffix : kSubNames) {
            char name[64];
            snprintf(name, sizeof(name), "layer%d_%s", i, suffix);
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (!t) continue;
            struct ggml_tensor * g = ggml_graph_get_grad(gf, t);
            if (g) {
                ggml_set_output(g);
            }
            sub_tensors.push_back(t);
            sub_grads.push_back(g);
            sub_labels.push_back(name);
        }
    }

    struct ggml_tensor * t_enc     = ggml_graph_get_tensor(gf, "enc_hidden");
    struct ggml_tensor * t_t       = ggml_graph_get_tensor(gf, "t");
    struct ggml_tensor * t_tr      = ggml_graph_get_tensor(gf, "t_r");
    struct ggml_tensor * t_pos     = ggml_graph_get_tensor(gf, "positions");
    struct ggml_tensor * t_sa_mask = ggml_graph_get_tensor(gf, "sa_mask_sw");
    struct ggml_tensor * t_ca_mask = ggml_graph_get_tensor(gf, "ca_mask");

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t->dit.backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "[GradCompare] FATAL: failed to allocate graph\n");
        exit(1);
    }

    struct ggml_tensor * loss_grad_acc = ggml_graph_get_grad_acc(gf, loss);
    {
        float onef = 1.0f;
        ggml_backend_tensor_set(loss_grad_acc, &onef, 0, sizeof(float));
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
    for (int i = 0; i < S; i++) pos_data[i] = i;
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

    enum ggml_status st = ggml_backend_graph_compute(t->dit.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[GradCompare] FATAL: graph compute failed (status=%d)\n", (int) st);
        exit(1);
    }

    float loss_val = 0.0f;
    ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));

    for (int i = 0; i < n_layers; i++) {
        struct ggml_tensor * grad = hidden_grads[i];
        if (!grad) {
            fprintf(stderr, "[GradCompare] WARNING: no gradient for hidden_after_layer%d\n", i);
            continue;
        }
        std::vector<float> g((size_t) ggml_nelements(grad));
        ggml_backend_tensor_get(grad, g.data(), 0, g.size() * sizeof(float));
        char fname[128];
        snprintf(fname, sizeof(fname), "/grad_hidden_layer%d.bin", i);
        write_raw(dump_dir + fname, g);
    }

    for (size_t i = 0; i < sub_tensors.size(); i++) {
        // Forward VALUE too (not just gradient) -- checks whether Q/K/V's
        // actual forward values match Python at this exact point, since
        // everything verified so far only checked the aggregate loss.
        if (sub_tensors[i]) {
            std::vector<float> fv((size_t) ggml_nelements(sub_tensors[i]));
            ggml_backend_tensor_get(sub_tensors[i], fv.data(), 0, fv.size() * sizeof(float));
            write_raw(dump_dir + "/fwd_" + sub_labels[i] + ".bin", fv);
        }
        if (!sub_grads[i]) {
            fprintf(stderr, "[GradCompare] WARNING: no gradient for %s\n", sub_labels[i].c_str());
            continue;
        }
        std::vector<float> g((size_t) ggml_nelements(sub_grads[i]));
        ggml_backend_tensor_get(sub_grads[i], g.data(), 0, g.size() * sizeof(float));
        write_raw(dump_dir + "/grad_" + sub_labels[i] + ".bin", g);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return loss_val;
}

int main(int argc, char ** argv) {
    std::string models_dir, sample_path, dump_dir;
    int         rank        = 64;
    int         alpha       = 128;
    float       t_val       = 0.7f;
    int         synthetic_T          = 0;  // 0 = use --smp; >0 = generate synthetic data instead
    int         synthetic_enc_S      = 8;
    bool        synth_replace_enc     = false;  // with --smp: keep real latents, swap in small synthetic enc_hidden
    bool        synth_replace_latents = false;  // with --smp: keep real enc_hidden, swap in small synthetic latents
    float       scale_enc             = 1.0f;   // with --smp: multiply real encoder_hidden by this (magnitude probe)
    int         max_layers            = 0;      // 0 = full model; cap for layer-depth bisection
    bool        dump_layer_grads      = false;  // dump d(loss)/d(hidden_after_layerN) for every N
    bool        finite_diff           = false;  // self-contained numerical-vs-analytic gradient check (no Python)

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--models" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--smp" && i + 1 < argc) {
            sample_path = argv[++i];
        } else if (arg == "--dump" && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (arg == "--rank" && i + 1 < argc) {
            rank = atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            alpha = atoi(argv[++i]);
        } else if (arg == "--t" && i + 1 < argc) {
            t_val = (float) atof(argv[++i]);
        } else if (arg == "--synthetic-T" && i + 1 < argc) {
            synthetic_T = atoi(argv[++i]);
        } else if (arg == "--synthetic-enc-s" && i + 1 < argc) {
            synthetic_enc_S = atoi(argv[++i]);
        } else if (arg == "--synth-replace-enc") {
            synth_replace_enc = true;
        } else if (arg == "--synth-replace-latents") {
            synth_replace_latents = true;
        } else if (arg == "--scale-enc" && i + 1 < argc) {
            scale_enc = (float) atof(argv[++i]);
        } else if (arg == "--layers" && i + 1 < argc) {
            max_layers = atoi(argv[++i]);
        } else if (arg == "--dump-layer-grads") {
            dump_layer_grads = true;
        } else if (arg == "--finite-diff") {
            finite_diff = true;
        }
    }
    if (models_dir.empty() || dump_dir.empty() || (sample_path.empty() && synthetic_T <= 0)) {
        fprintf(stderr,
                "Usage: %s --models <dir> --dump <dir> [--smp <gguf> | --synthetic-T N] [--rank N] [--alpha N] "
                "[--t F]\n",
                argv[0]);
        return 1;
    }
    std::filesystem::create_directories(dump_dir);

    ModelRegistry reg;
    if (!registry_scan(&reg, models_dir.c_str()) || reg.dit.empty()) {
        fprintf(stderr, "[GradCompare] FAIL: no DiT model found in %s\n", models_dir.c_str());
        return 1;
    }
    std::string dit_path = reg.dit[0].path;
    fprintf(stderr, "[GradCompare] Using DiT: %s\n", dit_path.c_str());

    DiTTrainSample smp;
    if (synthetic_T > 0) {
        smp.T     = synthetic_T;
        smp.enc_S = synthetic_enc_S;
        std::mt19937                     synth_rng(999);
        std::normal_distribution<float> synth_dist(0.0f, 1.0f);
        smp.target_latents.resize((size_t) smp.T * 64);
        for (float & v : smp.target_latents) v = synth_dist(synth_rng);
        smp.context_latents.assign((size_t) smp.T * 128, 0.0f);
        for (int ti = 0; ti < smp.T; ti++) {
            for (int c = 0; c < 64; c++) smp.context_latents[(size_t) ti * 128 + 64 + c] = 1.0f;
        }
        smp.encoder_hidden.resize((size_t) 2048 * smp.enc_S);
        for (float & v : smp.encoder_hidden) v = synth_dist(synth_rng) * 0.1f;
        fprintf(stderr, "[GradCompare] Synthetic sample: T=%d enc_S=%d\n", smp.T, smp.enc_S);
    } else if (!dit_train_sample_load(&smp, sample_path)) {
        fprintf(stderr, "[GradCompare] FAIL: could not load %s\n", sample_path.c_str());
        return 1;
    }
    fprintf(stderr, "[GradCompare] Sample: T=%d enc_S=%d\n", smp.T, smp.enc_S);

    if (synth_replace_enc) {
        std::mt19937                     r(999);
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (float & v : smp.encoder_hidden) v = d(r) * 0.1f;
        fprintf(stderr, "[GradCompare] Replaced encoder_hidden with small synthetic values (real latents kept)\n");
    }
    if (synth_replace_latents) {
        std::mt19937                     r(998);
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (float & v : smp.target_latents) v = d(r);
        smp.context_latents.assign((size_t) smp.T * 128, 0.0f);
        for (int ti = 0; ti < smp.T; ti++) {
            for (int c = 0; c < 64; c++) smp.context_latents[(size_t) ti * 128 + 64 + c] = 1.0f;
        }
        fprintf(stderr, "[GradCompare] Replaced target/context_latents with small synthetic values (real enc_hidden kept)\n");
    }
    if (scale_enc != 1.0f) {
        for (float & v : smp.encoder_hidden) v *= scale_enc;
        fprintf(stderr, "[GradCompare] Scaled real encoder_hidden by %.4f\n", scale_enc);
    }

    DiTTrainConfig cfg;
    cfg.rank  = rank;
    cfg.alpha = alpha;

    DiTTrain trainer;
    if (!dit_train_init(&trainer, dit_path.c_str(), cfg, max_layers, /*seed=*/42)) {
        fprintf(stderr, "[GradCompare] FAIL: dit_train_init failed\n");
        return 1;
    }

    // Overwrite layer0.sa_q.A with a fixed, exported (not RNG-reproduced)
    // pattern -- deterministic, small, and written to disk so Python loads
    // the *exact* same values rather than trying to replicate std::mt19937.
    struct ggml_tensor * A = trainer.lora[0].sa_q.A;
    int64_t                a_n = ggml_nelements(A);
    std::vector<float>     a_vals((size_t) a_n);
    std::mt19937            a_rng(1234);
    std::uniform_real_distribution<float> a_dist(-0.02f, 0.02f);
    for (float & v : a_vals) {
        v = a_dist(a_rng);
    }
    ggml_backend_tensor_set(A, a_vals.data(), 0, a_n * sizeof(float));
    write_raw(dump_dir + "/A_init.bin", a_vals);

    // B stays at its standard zero init (already zero from dit_train_init),
    // *unless* doing a finite-diff check: B=0 makes dL/dA identically zero
    // (see dit_lora_delta -- delta=scale*B@(A@x), so dL/dA depends on B),
    // which is useless for numerically verifying the A gradient. Give B a
    // small nonzero value first so both A and B have a meaningful gradient
    // to check, matching the "mid-training" regime real training spends
    // almost all its time in (B is only exactly zero at step 0).
    struct ggml_tensor * B = trainer.lora[0].sa_q.B;
    int64_t                b_n = ggml_nelements(B);
    std::vector<float>     b_vals((size_t) b_n, 0.0f);
    if (finite_diff) {
        std::mt19937                     b_rng(4321);
        std::uniform_real_distribution<float> b_dist(-0.02f, 0.02f);
        for (float & v : b_vals) v = b_dist(b_rng);
        ggml_backend_tensor_set(B, b_vals.data(), 0, b_n * sizeof(float));
    }
    write_raw(dump_dir + "/B_init.bin", b_vals);

    // Fixed noise (x1), exported so Python uses the identical array.
    std::vector<float>     noise((size_t) smp.T * 64);
    std::mt19937            n_rng(5678);
    std::normal_distribution<float> n_dist(0.0f, 1.0f);
    for (float & v : noise) {
        v = n_dist(n_rng);
    }
    write_raw(dump_dir + "/noise.bin", noise);
    write_raw(dump_dir + "/target_latents.bin", smp.target_latents);
    write_raw(dump_dir + "/context_latents.bin", smp.context_latents);
    write_raw(dump_dir + "/encoder_hidden.bin", smp.encoder_hidden);

    {
        FILE * f = fopen((dump_dir + "/meta.txt").c_str(), "w");
        fprintf(f, "T=%d\nenc_S=%d\nrank=%d\nalpha=%d\nt=%.8f\n", smp.T, smp.enc_S, rank, alpha, t_val);
        fclose(f);
    }

    if (finite_diff) {
        // Self-contained check: no Python involved. Compute the analytic
        // gradient for a few of layer0.sa_q's A elements via the real
        // training path (forward_backward), then numerically estimate the
        // same gradients via central finite differences using ONLY
        // forward-pass evaluations (dit_train_eval) of the *same* trainer/
        // graph-building code. If these disagree, the bug is proven to be
        // in our own backward implementation -- no cross-language ambiguity.
        float loss0 = dit_train_forward_backward(&trainer, smp.T, smp.enc_S, smp.target_latents.data(),
                                                  smp.context_latents.data(), smp.encoder_hidden.data(),
                                                  noise.data(), t_val);
        std::vector<float> analytic_gA = trainer.grad_accum[0].sa_q.A;  // accum_count=1, this call's raw gradient
        fprintf(stderr, "[FiniteDiff] loss=%.8f\n", loss0);

        int64_t in_feat = A->ne[0];
        int     n_check = 6;
        float   eps     = 0.5f;
        fprintf(stderr, "[FiniteDiff] %8s %16s %16s %10s\n", "A_idx", "analytic", "numerical", "rel_err");
        for (int k = 0; k < n_check; k++) {
            int64_t idx = (int64_t) k * (in_feat / n_check);  // spread across the first LoRA rank row

            std::vector<float> a_plus = a_vals;
            a_plus[idx] += eps;
            ggml_backend_tensor_set(A, a_plus.data(), 0, a_plus.size() * sizeof(float));
            float loss_plus = dit_train_eval(&trainer, smp.T, smp.enc_S, smp.target_latents.data(),
                                             smp.context_latents.data(), smp.encoder_hidden.data(), noise.data(),
                                             t_val);

            std::vector<float> a_minus = a_vals;
            a_minus[idx] -= eps;
            ggml_backend_tensor_set(A, a_minus.data(), 0, a_minus.size() * sizeof(float));
            float loss_minus = dit_train_eval(&trainer, smp.T, smp.enc_S, smp.target_latents.data(),
                                              smp.context_latents.data(), smp.encoder_hidden.data(), noise.data(),
                                              t_val);

            ggml_backend_tensor_set(A, a_vals.data(), 0, a_vals.size() * sizeof(float));  // restore

            double numeric  = ((double) loss_plus - (double) loss_minus) / (2.0 * (double) eps);
            double analytic = (double) analytic_gA[(size_t) idx];
            double rel_err  = std::fabs(numeric - analytic) / (std::fabs(analytic) + std::fabs(numeric) + 1e-12);
            fprintf(stderr, "[FiniteDiff] %8lld %16.8e %16.8e %10.4f\n", (long long) idx, analytic, numeric,
                    rel_err);
        }
        dit_train_free(&trainer);
        return 0;
    }

    float loss;
    if (dump_layer_grads) {
        loss = dit_train_forward_backward_layer_grads(&trainer, smp.T, smp.enc_S, smp.target_latents.data(),
                                                       smp.context_latents.data(), smp.encoder_hidden.data(),
                                                       noise.data(), t_val, dump_dir);
    } else {
        loss = dit_train_forward_backward(&trainer, smp.T, smp.enc_S, smp.target_latents.data(),
                                          smp.context_latents.data(), smp.encoder_hidden.data(), noise.data(),
                                          t_val);
    }
    fprintf(stderr, "[GradCompare] loss=%.8f\n", loss);

    write_raw(dump_dir + "/loss.bin", { loss });
    if (!dump_layer_grads) {
        write_raw(dump_dir + "/grad_A.bin", trainer.grad_accum[0].sa_q.A);
        write_raw(dump_dir + "/grad_B.bin", trainer.grad_accum[0].sa_q.B);
    }

    fprintf(stderr, "[GradCompare] Dumped to %s (A_init, B_init, noise, target_latents, context_latents, "
                    "encoder_hidden, loss, grad_A, grad_B)\n",
            dump_dir.c_str());

    dit_train_free(&trainer);
    return 0;
}
