// ace-train.cpp: native LoRA training for the ACE-Step DiT (Phase 3, see
// ../TRAINING_DEV.md). Two subcommands:
//
//   ace-train fit     --models <dir> --tensors <dir> --output <dir> [...]
//   ace-train prepare --models <dir> --dataset <dir> --output <dir> [...]
//
// `fit` trains LoRA adapters from a directory of preprocessed per-sample
// tensor GGUFs (src/dit-train-data.h). `prepare` (dataset decoration via the
// understand pipeline + VAE/text/cond encoding) is Phase 4 and not yet
// implemented -- this binary reports that clearly rather than guessing.

#include "dit-train-data.h"
#include "dit-train.h"
#include "model-registry.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct FitArgs {
    std::string models_dir;
    std::string tensors_dir;
    std::string output_dir;
    std::string dit_model;  // optional filename override, else first DiT in registry

    int   rank            = 8;
    int   alpha            = 16;
    float lr                = 1e-4f;
    float weight_decay      = 0.01f;
    int   epochs            = 100;
    int   grad_accum        = 4;
    int   save_every        = 10;
    uint32_t seed           = 42;
    float val_split         = 0.0f;
    int   warmup_steps      = 100;
    int   max_layers        = 0;  // 0 = full model; debug-only cap
    int   log_every         = 10;
};

static void usage_fit(const char * prog) {
    fprintf(stderr,
            "acestep.cpp %s\n\n"
            "Usage: %s fit --models <dir> --tensors <dir> --output <dir> [options]\n"
            "\n"
            "Required:\n"
            "  --models <dir>          Directory of GGUF model files (needs a DiT)\n"
            "  --tensors <dir>         Directory of preprocessed sample tensor GGUFs\n"
            "                          (src/dit-train-data.h; see ace-train prepare)\n"
            "  --output <dir>          Where to write adapter checkpoints\n"
            "\n"
            "LoRA:\n"
            "  --rank <N>              LoRA rank (default: 8)\n"
            "  --alpha <N>             LoRA alpha (default: 16)\n"
            "\n"
            "Training:\n"
            "  --lr <F>                Learning rate (default: 1e-4)\n"
            "  --weight-decay <F>       AdamW weight decay (default: 0.01)\n"
            "  --epochs <N>             Epochs over the training set (default: 100)\n"
            "  --grad-accum <N>         Micro-batches per optimizer step (default: 4)\n"
            "  --warmup-steps <N>       LR warmup steps, capped to total/10 (default: 100)\n"
            "  --val-split <F>          Fraction of samples held out for validation (default: 0)\n"
            "  --seed <N>               RNG seed (default: 42)\n"
            "\n"
            "Checkpoints:\n"
            "  --save-every <N>         Save a checkpoint every N epochs (default: 10)\n"
            "\n"
            "Debug:\n"
            "  --dit-model <name>       DiT GGUF filename (default: first DiT found)\n"
            "  --max-layers <N>         Cap the model to its first N layers\n"
            "  --log-every <N>          Log every N optimizer steps (default: 10)\n",
            ACE_VERSION, prog);
}

// LinearLR warmup (0.1x -> 1.0x base_lr) then single-cycle cosine annealing
// down to base_lr * 0.01. Matches the Python trainer's scheduler exactly
// (CosineAnnealingWarmRestarts with T_0 = total-warmup, T_mult=1, i.e. one
// cosine half-cycle since we never run past total_steps).
static float dit_train_lr_schedule(int64_t step, int64_t total_steps, float base_lr, int64_t warmup_steps) {
    if (warmup_steps > 0 && step < warmup_steps) {
        float frac = (float) step / (float) warmup_steps;
        return base_lr * (0.1f + 0.9f * frac);
    }
    int64_t remaining = total_steps - warmup_steps;
    if (remaining <= 0) {
        return base_lr;
    }
    float   eta_min = base_lr * 0.01f;
    int64_t t       = step - warmup_steps;
    float   frac    = (float) t / (float) remaining;
    return eta_min + 0.5f * (base_lr - eta_min) * (1.0f + cosf((float) M_PI * frac));
}

static int run_fit(const char * prog, int argc, char ** argv) {
    FitArgs a;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        auto        need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "[ace-train] FATAL: %s needs a value\n", name);
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "--models") {
            a.models_dir = need("--models");
        } else if (arg == "--tensors") {
            a.tensors_dir = need("--tensors");
        } else if (arg == "--output") {
            a.output_dir = need("--output");
        } else if (arg == "--dit-model") {
            a.dit_model = need("--dit-model");
        } else if (arg == "--rank") {
            a.rank = atoi(need("--rank"));
        } else if (arg == "--alpha") {
            a.alpha = atoi(need("--alpha"));
        } else if (arg == "--lr") {
            a.lr = (float) atof(need("--lr"));
        } else if (arg == "--weight-decay") {
            a.weight_decay = (float) atof(need("--weight-decay"));
        } else if (arg == "--epochs") {
            a.epochs = atoi(need("--epochs"));
        } else if (arg == "--grad-accum") {
            a.grad_accum = atoi(need("--grad-accum"));
        } else if (arg == "--warmup-steps") {
            a.warmup_steps = atoi(need("--warmup-steps"));
        } else if (arg == "--val-split") {
            a.val_split = (float) atof(need("--val-split"));
        } else if (arg == "--seed") {
            a.seed = (uint32_t) atoi(need("--seed"));
        } else if (arg == "--save-every") {
            a.save_every = atoi(need("--save-every"));
        } else if (arg == "--max-layers") {
            a.max_layers = atoi(need("--max-layers"));
        } else if (arg == "--log-every") {
            a.log_every = atoi(need("--log-every"));
        } else if (arg == "-h" || arg == "--help") {
            usage_fit(prog);
            return 0;
        } else {
            fprintf(stderr, "[ace-train] FATAL: unknown argument '%s'\n", arg.c_str());
            usage_fit(prog);
            return 1;
        }
    }

    if (a.models_dir.empty() || a.tensors_dir.empty() || a.output_dir.empty()) {
        fprintf(stderr, "[ace-train] FATAL: --models, --tensors and --output are all required\n");
        usage_fit(prog);
        return 1;
    }

    ModelRegistry reg;
    if (!registry_scan(&reg, a.models_dir.c_str()) || reg.dit.empty()) {
        fprintf(stderr, "[ace-train] FATAL: no DiT model found in %s\n", a.models_dir.c_str());
        return 1;
    }
    std::string dit_path;
    if (!a.dit_model.empty()) {
        const ModelEntry * e = registry_find(reg.dit, a.dit_model.c_str());
        if (!e) {
            fprintf(stderr, "[ace-train] FATAL: DiT model '%s' not found in %s\n", a.dit_model.c_str(),
                    a.models_dir.c_str());
            return 1;
        }
        dit_path = e->path;
    } else {
        dit_path = reg.dit[0].path;
    }
    fprintf(stderr, "[ace-train] DiT: %s\n", dit_path.c_str());

    // ---- Load the tensor cache -------------------------------------------
    std::vector<std::string> sample_paths = dit_train_scan_tensor_dir(a.tensors_dir);
    if (sample_paths.empty()) {
        fprintf(stderr, "[ace-train] FATAL: no *.gguf samples found in %s\n", a.tensors_dir.c_str());
        return 1;
    }
    std::vector<DiTTrainSample> samples(sample_paths.size());
    for (size_t i = 0; i < sample_paths.size(); i++) {
        if (!dit_train_sample_load(&samples[i], sample_paths[i])) {
            fprintf(stderr, "[ace-train] FATAL: failed to load sample %s\n", sample_paths[i].c_str());
            return 1;
        }
    }
    fprintf(stderr, "[ace-train] Loaded %zu samples from %s\n", samples.size(), a.tensors_dir.c_str());

    // ---- Train/val split --------------------------------------------------
    std::vector<int> order(samples.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = (int) i;
    }
    std::mt19937 split_rng(a.seed);
    std::shuffle(order.begin(), order.end(), split_rng);

    size_t           n_val = (size_t) ((double) samples.size() * a.val_split);
    std::vector<int> val_idx(order.begin(), order.begin() + (long) n_val);
    std::vector<int> train_idx(order.begin() + (long) n_val, order.end());
    if (train_idx.empty()) {
        fprintf(stderr, "[ace-train] FATAL: val-split leaves no training samples\n");
        return 1;
    }
    fprintf(stderr, "[ace-train] Train: %zu, Val: %zu\n", train_idx.size(), val_idx.size());

    // ---- Init trainer -------------------------------------------------
    DiTTrainConfig cfg;
    cfg.rank          = a.rank;
    cfg.alpha         = a.alpha;
    cfg.lr            = a.lr;
    cfg.weight_decay  = a.weight_decay;

    DiTTrain trainer;
    if (!dit_train_init(&trainer, dit_path.c_str(), cfg, a.max_layers, a.seed)) {
        fprintf(stderr, "[ace-train] FATAL: dit_train_init failed\n");
        return 1;
    }

    // ---- Schedule -----------------------------------------------------
    int64_t steps_per_epoch = (int64_t) ((train_idx.size() + (size_t) a.grad_accum - 1) / (size_t) a.grad_accum);
    int64_t total_steps     = steps_per_epoch * a.epochs;
    int64_t warmup_steps    = std::min<int64_t>(a.warmup_steps, std::max<int64_t>(1, total_steps / 10));
    fprintf(stderr, "[ace-train] %d epochs x %lld steps/epoch = %lld total optimizer steps (warmup=%lld)\n",
            a.epochs, (long long) steps_per_epoch, (long long) total_steps, (long long) warmup_steps);

    std::mt19937                       train_rng(a.seed + 1);
    std::mt19937                       data_rng(a.seed + 2);
    std::normal_distribution<float>    noise_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> t_idx_dist(0, 7);

    int64_t global_step  = 0;
    double  ema_loss      = -1.0;
    const double ema_alpha = 0.1;
    double  best_val_loss  = 1e30;

    for (int epoch = 0; epoch < a.epochs; epoch++) {
        std::shuffle(train_idx.begin(), train_idx.end(), train_rng);

        size_t i = 0;
        while (i < train_idx.size()) {
            size_t batch_end = std::min(train_idx.size(), i + (size_t) a.grad_accum);
            double batch_loss = 0.0;
            int    batch_n     = 0;

            for (size_t k = i; k < batch_end; k++) {
                const DiTTrainSample & s = samples[(size_t) train_idx[k]];
                std::vector<float>     noise((size_t) s.T * 64);
                for (float & v : noise) {
                    v = noise_dist(data_rng);
                }
                float t_val = DIT_TRAIN_TURBO_SHIFT3_TIMESTEPS[t_idx_dist(data_rng)];

                float loss = dit_train_forward_backward(&trainer, s.T, s.enc_S, s.target_latents.data(),
                                                        s.context_latents.data(), s.encoder_hidden.data(),
                                                        noise.data(), t_val);
                batch_loss += loss;
                batch_n++;
            }

            trainer.cfg.lr = dit_train_lr_schedule(global_step, total_steps, a.lr, warmup_steps);
            dit_train_optimizer_step(&trainer);
            global_step++;

            batch_loss /= std::max(batch_n, 1);
            ema_loss = (ema_loss < 0.0) ? batch_loss : (ema_alpha * batch_loss + (1.0 - ema_alpha) * ema_loss);
            if (global_step % a.log_every == 0) {
                fprintf(stderr, "[ace-train] epoch %d/%d step %lld/%lld lr=%.2e loss=%.6f ema=%.6f\n", epoch + 1,
                        a.epochs, (long long) global_step, (long long) total_steps, trainer.cfg.lr, batch_loss,
                        ema_loss);
            }

            i = batch_end;
        }

        if (!val_idx.empty()) {
            double val_loss = 0.0;
            for (int vi : val_idx) {
                const DiTTrainSample & s = samples[(size_t) vi];
                std::vector<float>     noise((size_t) s.T * 64);
                for (float & v : noise) {
                    v = noise_dist(data_rng);
                }
                float t_val = DIT_TRAIN_TURBO_SHIFT3_TIMESTEPS[t_idx_dist(data_rng)];
                val_loss += dit_train_eval(&trainer, s.T, s.enc_S, s.target_latents.data(), s.context_latents.data(),
                                          s.encoder_hidden.data(), noise.data(), t_val);
            }
            val_loss /= (double) val_idx.size();
            fprintf(stderr, "[ace-train] epoch %d/%d val_loss=%.6f\n", epoch + 1, a.epochs, val_loss);
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                dit_train_save_checkpoint(&trainer, a.output_dir + "/checkpoints/best");
            }
        }

        if ((epoch + 1) % a.save_every == 0) {
            std::string dir = a.output_dir + "/checkpoints/epoch_" + std::to_string(epoch + 1);
            dit_train_save_checkpoint(&trainer, dir);
        }
    }

    dit_train_save_checkpoint(&trainer, a.output_dir + "/final");
    fprintf(stderr, "[ace-train] Done. Final EMA loss: %.6f\n", ema_loss);

    dit_train_free(&trainer);
    return 0;
}

static int run_prepare(int /*argc*/, char ** /*argv*/) {
    fprintf(stderr,
            "[ace-train] 'prepare' is not implemented yet (Phase 4, see TRAINING_DEV.md).\n"
            "Build the tensor cache by hand for now: src/dit-train-data.h exposes\n"
            "dit_train_sample_write() for that (see tests/test-lora-train-smoke.cpp for\n"
            "the tensor shapes/conventions expected by 'ace-train fit').\n");
    return 1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <fit|prepare> [options]\n", argv[0]);
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "fit") {
        return run_fit(argv[0], argc - 2, argv + 2);
    }
    if (cmd == "prepare") {
        return run_prepare(argc - 2, argv + 2);
    }
    fprintf(stderr, "[ace-train] FATAL: unknown subcommand '%s' (expected 'fit' or 'prepare')\n", cmd.c_str());
    return 1;
}
