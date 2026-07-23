// test-lora-train-smoke.cpp: LoRA training backward smoke test + overfit
// check (Phase 2 of native LoRA training, see ../TRAINING_DEV.md).
//
// Validates that the GGML autograd/ggml-opt machinery correctly covers the
// full LoRA-augmented DiT forward graph:
//   1. Builds the training graph (forward + flow-matching MSE loss +
//      backward + per-parameter AdamW step) and runs it. If any op on the
//      backward path lacks a GGML backward implementation, this aborts with
//      a clear "unsupported ggml op for backward pass" message -- exactly
//      the failure mode this test exists to catch.
//   2. Overfits a single fixed synthetic sample over many steps and checks
//      the loss drops substantially, proving gradients actually flow into
//      the LoRA A/B parameters (not just "the graph runs without crashing").
//   3. Confirms the frozen backbone is untouched (spot-check a weight
//      tensor's bytes before/after) and that at least one LoRA B tensor
//      moved off its zero init (proving the optimizer step landed).
//
// Usage:
//   ./test-lora-train-smoke --models <dir> [--layers N] [--steps N]
//                            [--rank N] [--alpha N] [--lr F] [--seed N]
//                            [--T N] [--enc-s N]
//
// All conditioning/target data is synthetic (random, fixed for the whole
// run): this test is about autograd correctness, not audio quality. Real
// dataset preprocessing is Phase 4.

#include "dit-train.h"
#include "model-registry.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    std::string models_dir;
    int         max_layers = 0;
    int         steps      = 60;
    int         rank       = 8;
    int         alpha      = 16;
    float       lr         = 1e-3f;  // higher than the 1e-4 production default: this is a short overfit smoke test
    uint32_t    seed       = 42;
    int         T          = 32;
    int         enc_S      = 8;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto        next_i = [&]() { return atoi(argv[++i]); };
        auto        next_f = [&]() { return (float) atof(argv[++i]); };
        if (arg == "--models" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--layers" && i + 1 < argc) {
            max_layers = next_i();
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = next_i();
        } else if (arg == "--rank" && i + 1 < argc) {
            rank = next_i();
        } else if (arg == "--alpha" && i + 1 < argc) {
            alpha = next_i();
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = next_f();
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = (uint32_t) next_i();
        } else if (arg == "--T" && i + 1 < argc) {
            T = next_i();
        } else if (arg == "--enc-s" && i + 1 < argc) {
            enc_S = next_i();
        }
    }
    if (models_dir.empty()) {
        fprintf(stderr,
                "Usage: %s --models <dir> [--layers N] [--steps N] [--rank N] [--alpha N] [--lr F] [--seed N] "
                "[--T N] [--enc-s N]\n",
                argv[0]);
        return 1;
    }

    ModelRegistry reg;
    if (!registry_scan(&reg, models_dir.c_str()) || reg.dit.empty()) {
        fprintf(stderr, "[Test] FAIL: no DiT model found in %s\n", models_dir.c_str());
        return 1;
    }
    std::string dit_path = reg.dit[0].path;
    fprintf(stderr, "[Test] Using DiT: %s\n", dit_path.c_str());

    DiTTrainConfig cfg;
    cfg.rank  = rank;
    cfg.alpha = alpha;
    cfg.lr    = lr;

    DiTTrain trainer;
    if (!dit_train_init(&trainer, dit_path.c_str(), cfg, max_layers, seed)) {
        fprintf(stderr, "[Test] FAIL: dit_train_init failed\n");
        return 1;
    }

    int Oc    = trainer.dit.cfg.out_channels;
    int H_enc = (int) trainer.dit.cond_emb_w->ne[0];
    if (T % trainer.dit.cfg.patch_size != 0) {
        T = (T / trainer.dit.cfg.patch_size + 1) * trainer.dit.cfg.patch_size;
        fprintf(stderr, "[Test] Rounded T up to %d (multiple of patch_size=%d)\n", T, trainer.dit.cfg.patch_size);
    }

    fprintf(stderr, "[Test] T=%d, enc_S=%d, Oc=%d, H_enc=%d, steps=%d, rank=%d, alpha=%d, lr=%.2e\n", T, enc_S, Oc,
            H_enc, steps, rank, alpha, lr);

    // ---- One fixed synthetic sample, overfit across all steps -------------
    std::mt19937                     data_rng(seed + 1);
    std::normal_distribution<float> latent_dist(0.0f, 1.0f);

    std::vector<float> target_latents((size_t) T * Oc);  // x0
    for (float & v : target_latents) {
        v = latent_dist(data_rng);
    }

    // text2music context: silence (zeros) + mask=1.0, matches
    // build_context_latents() in the Python preprocessing reference.
    std::vector<float> context_latents((size_t) T * 2 * Oc, 0.0f);
    for (int ti = 0; ti < T; ti++) {
        for (int c = 0; c < Oc; c++) {
            context_latents[(size_t) ti * 2 * Oc + Oc + c] = 1.0f;  // mask half = 1.0
        }
    }

    std::vector<float> enc_hidden((size_t) H_enc * enc_S);
    for (float & v : enc_hidden) {
        v = latent_dist(data_rng) * 0.1f;
    }

    // Fixed noise + timestep: this is a classic single-example overfit test
    // (loss -> ~0), so (x0, x1, t) together form ONE regression target held
    // constant across every step. Real training resamples x1/t per pass
    // instead (see tools/ace-train.cpp's fit loop, Phase 3).
    std::mt19937                       data_rng2(seed + 2);
    std::uniform_int_distribution<int> t_idx_dist(0, 7);
    std::vector<float>                 noise((size_t) T * Oc);
    for (float & v : noise) {
        v = latent_dist(data_rng2);
    }
    float t_val = DIT_TRAIN_TURBO_SHIFT3_TIMESTEPS[t_idx_dist(data_rng2)];
    fprintf(stderr, "[Test] Fixed overfit target: t=%.4f\n", t_val);

    // ---- Spot-check setup: frozen weight bytes + a LoRA B tensor ----------
    struct ggml_tensor * frozen_probe = trainer.dit.layers[0].sa_q_proj;
    size_t                probe_n     = (size_t) ggml_nelements(frozen_probe);
    std::vector<float>    frozen_before(probe_n), frozen_after(probe_n);
    ggml_backend_tensor_get(frozen_probe, frozen_before.data(), 0, probe_n * sizeof(float));

    struct ggml_tensor * lora_b_probe = trainer.lora[0].sa_q.B;
    size_t                b_n         = (size_t) ggml_nelements(lora_b_probe);
    std::vector<float>    b_before(b_n), b_after(b_n);
    ggml_backend_tensor_get(lora_b_probe, b_before.data(), 0, b_n * sizeof(float));

    // ---- Training loop ------------------------------------------------
    std::vector<float> losses;
    losses.reserve(steps);

    for (int step = 0; step < steps; step++) {
        float loss = dit_train_forward_backward(&trainer, T, enc_S, target_latents.data(), context_latents.data(),
                                                enc_hidden.data(), noise.data(), t_val);
        dit_train_optimizer_step(&trainer);
        losses.push_back(loss);

        if (!std::isfinite(loss)) {
            fprintf(stderr, "[Test] FAIL: non-finite loss at step %d\n", step);
            return 1;
        }
        if (step % 5 == 0 || step == steps - 1) {
            fprintf(stderr, "[Test] step %3d  t=%.4f  loss=%.6f\n", step, t_val, loss);
        }
    }

    ggml_backend_tensor_get(frozen_probe, frozen_after.data(), 0, probe_n * sizeof(float));
    ggml_backend_tensor_get(lora_b_probe, b_after.data(), 0, b_n * sizeof(float));

    // ---- Verification ---------------------------------------------------
    int n_failed = 0;

    // 1) frozen backbone must be bit-identical
    if (memcmp(frozen_before.data(), frozen_after.data(), probe_n * sizeof(float)) != 0) {
        fprintf(stderr, "[Test] FAIL: frozen backbone weight changed during training\n");
        n_failed++;
    } else {
        fprintf(stderr, "[Test] OK: frozen backbone weight untouched\n");
    }

    // 2) at least one LoRA B tensor must have moved off its zero init
    double b_delta = 0.0;
    for (size_t i = 0; i < b_n; i++) {
        b_delta += std::fabs((double) b_after[i] - (double) b_before[i]);
    }
    if (b_delta < 1e-9) {
        fprintf(stderr, "[Test] FAIL: LoRA B (layer 0, sa_q) did not move from its zero init\n");
        n_failed++;
    } else {
        fprintf(stderr, "[Test] OK: LoRA B (layer 0, sa_q) moved (sum |delta|=%.6f)\n", b_delta);
    }

    // 3) loss must trend down: compare mean of the first 5 vs last 5 steps
    int    edge      = std::min(5, steps);
    double first_avg = 0.0, last_avg = 0.0;
    for (int i = 0; i < edge; i++) {
        first_avg += losses[i];
    }
    for (int i = steps - edge; i < steps; i++) {
        last_avg += losses[i];
    }
    first_avg /= edge;
    last_avg /= edge;
    fprintf(stderr, "[Test] loss avg: first %d steps=%.6f, last %d steps=%.6f\n", edge, first_avg, edge, last_avg);
    if (!(last_avg < first_avg * 0.7)) {
        fprintf(stderr, "[Test] FAIL: loss did not drop substantially (expected last < 0.7 * first)\n");
        n_failed++;
    } else {
        fprintf(stderr, "[Test] OK: loss dropped (%.6f -> %.6f)\n", first_avg, last_avg);
    }

    dit_train_free(&trainer);

    if (n_failed > 0) {
        fprintf(stderr, "[Test] RESULT: FAIL (%d check(s) failed)\n", n_failed);
        return 1;
    }
    fprintf(stderr, "[Test] RESULT: PASS\n");
    return 0;
}
