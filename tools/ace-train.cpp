// ace-train.cpp: native LoRA training for the ACE-Step DiT (Phases 3 + 4,
// see ../TRAINING_DEV.md). Two subcommands:
//
//   ace-train fit     --models <dir> --tensors <dir> --output <dir> [...]
//   ace-train prepare --models <dir> --dataset <dir> --output <dir> [...]
//
// `fit` trains LoRA adapters from a directory of preprocessed per-sample
// tensor GGUFs (src/dit-train-data.h). `prepare` scans a directory of audio
// files + sidecar labels, auto-labels missing captions via the understand
// pipeline, and encodes each into that tensor cache (src/dit-prepare.h).

#include "dit-prepare.h"
#include "dit-train-data.h"
#include "dit-train.h"
#include "model-registry.h"
#include "model-store.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
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
    float cfg_ratio         = 0.15f;  // CFG dropout probability (0 = disabled)
    float max_grad_norm     = 1.0f;  // 0 = disabled; matches Python trainer's default of 1.0
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
            "  --cfg-ratio <F>          CFG dropout probability, condition -> null embedding\n"
            "                          (default: 0.15, matches the Python reference's corrected\n"
            "                          trainer; 0 disables)\n"
            "  --max-grad-norm <F>      Global gradient-norm clip across all LoRA tensors\n"
            "                          jointly, applied before the AdamW step (default: 1.0,\n"
            "                          matches the Python reference; 0 disables)\n"
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
        } else if (arg == "--max-grad-norm") {
            a.max_grad_norm = (float) atof(need("--max-grad-norm"));
        } else if (arg == "--cfg-ratio") {
            a.cfg_ratio = (float) atof(need("--cfg-ratio"));
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
    cfg.max_grad_norm = a.max_grad_norm;

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

    std::mt19937                    train_rng(a.seed + 1);
    std::mt19937                    data_rng(a.seed + 2);
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    // Continuous logit-normal timestep sampling: t = sigmoid(N(mu, sigma)),
    // mu=-0.4 sigma=1.0 for every ACE-Step model variant (turbo/base/sft
    // all share these -- "shift" is an inference-only step-schedule
    // parameter, never applied during training). This is a faithful port
    // of the model's own sample_t_r() (see acestep-repo's
    // training_v2/timestep_sampling.py, itself reimplementing
    // modeling_acestep_v15_turbo.py) -- NOT the discrete 8-value
    // DIT_TRAIN_TURBO_SHIFT3_TIMESTEPS schedule used until now, which the
    // Python project's own CLI labels "vanilla (bugged)": training on only
    // 8 fixed points starves the LoRA of the full continuous timestep
    // range, so it never learns a smooth, generalizable correction. See
    // TRAINING_DEV.md.
    // Python's sample_timesteps() (training_v2/timestep_sampling.py) draws
    // TWO independent sigmoid(N(mu,sigma)) samples per step and assigns
    // t=max, r=min -- then forces r=t anyway (data_proportion=1.0 for
    // every ACE-Step variant during training, since use_meanflow=False).
    // Net effect: r is discarded, but t itself is the MAX of two draws,
    // not a single draw -- skews the effective t distribution higher on
    // average than a single sigmoid(N(mu,sigma)) sample would. Match it
    // exactly rather than approximate, since "max of 2" is not the same
    // distribution as "1 draw with a different mu".
    std::normal_distribution<float> t_logit_dist(-0.4f, 1.0f);
    auto                             sample_t = [&](std::mt19937 & rng) {
        float t1 = 1.0f / (1.0f + std::exp(-t_logit_dist(rng)));
        float t2 = 1.0f / (1.0f + std::exp(-t_logit_dist(rng)));
        return std::max(t1, t2);
    };

    // CFG dropout: with probability a.cfg_ratio, replace this step's real
    // conditioning with the model's own null_condition_emb (broadcast
    // across every encoder position), matching Python's apply_cfg_dropout.
    // Without this, the LoRA-modified weights never see the unconditional
    // branch that real CFG-guided generation also evaluates.
    std::uniform_real_distribution<float> cfg_dropout_dist(0.0f, 1.0f);
    std::vector<float>                    null_cond_vec;
    if (trainer.dit.null_condition_emb) {
        struct ggml_tensor * nce   = trainer.dit.null_condition_emb;
        int64_t               emb_n = ggml_nelements(nce);
        null_cond_vec.resize((size_t) emb_n);
        if (nce->type == GGML_TYPE_BF16) {
            std::vector<uint16_t> raw((size_t) emb_n);
            ggml_backend_tensor_get(nce, raw.data(), 0, (size_t) emb_n * sizeof(uint16_t));
            for (int64_t j = 0; j < emb_n; j++) {
                uint32_t w = (uint32_t) raw[(size_t) j] << 16;
                memcpy(&null_cond_vec[(size_t) j], &w, 4);
            }
        } else if (nce->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(nce, null_cond_vec.data(), 0, (size_t) emb_n * sizeof(float));
        } else {
            fprintf(stderr, "[ace-train] WARNING: null_condition_emb unexpected type %d, CFG dropout disabled\n",
                    (int) nce->type);
            null_cond_vec.clear();
        }
    }
    bool have_cfg_dropout = !null_cond_vec.empty() && a.cfg_ratio > 0.0f;
    fprintf(stderr, "[ace-train] CFG dropout: %s (ratio=%.2f)\n", have_cfg_dropout ? "enabled" : "disabled",
            a.cfg_ratio);

    // Scratch buffer for a CFG-dropped sample's encoder_hidden: null_cond_vec
    // (one H_enc-sized vector) tiled across every encoder position.
    std::vector<float> null_enc_hidden;

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
                float t_val = sample_t(data_rng);

                const float * enc_hidden_data = s.encoder_hidden.data();
                if (have_cfg_dropout && cfg_dropout_dist(data_rng) < a.cfg_ratio) {
                    int64_t H_enc = (int64_t) s.encoder_hidden.size() / s.enc_S;
                    null_enc_hidden.resize((size_t) H_enc * s.enc_S);
                    for (int j = 0; j < s.enc_S; j++) {
                        memcpy(&null_enc_hidden[(size_t) j * H_enc], null_cond_vec.data(),
                               (size_t) H_enc * sizeof(float));
                    }
                    enc_hidden_data = null_enc_hidden.data();
                }

                float loss = dit_train_forward_backward(&trainer, s.T, s.enc_S, s.target_latents.data(),
                                                        s.context_latents.data(), enc_hidden_data, noise.data(),
                                                        t_val);
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
                float t_val = sample_t(data_rng);
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

struct PrepareArgs {
    std::string models_dir;
    std::string dataset_dir;
    std::string output_dir;
    std::string dit_model;
    std::string text_enc_model;
    std::string vae_model;
    std::string lm_model;   // optional, enables auto-labeling missing captions
    float       max_duration  = 240.0f;
    int         vae_chunk     = 1024;
    int         vae_overlap   = 64;
    bool        skip_existing = false;
    std::string trigger_word;
    std::string tag_position = "prepend";
};

static void usage_prepare(const char * prog) {
    fprintf(stderr,
            "acestep.cpp %s\n\n"
            "Usage: %s prepare --models <dir> --dataset <dir> --output <dir> [options]\n"
            "\n"
            "Required:\n"
            "  --models <dir>          Directory of GGUF model files (needs DiT, Text-Enc, VAE)\n"
            "  --dataset <dir>         Directory of audio files + sidecar labels\n"
            "                          ({name}.json, {name}.lyrics.txt/.txt, {name}.caption.txt)\n"
            "  --output <dir>          Where to write sample tensor GGUFs\n"
            "\n"
            "Auto-labeling:\n"
            "  --lm-model <name>       LM GGUF filename; fills captions missing from sidecars\n"
            "                          (samples with no caption and no --lm-model are skipped)\n"
            "\n"
            "Trigger word (recommended for style/single-song adapters -- see TRAINING_DEV.md):\n"
            "  --trigger-word <word>   Short tag applied to every sample's caption, so the whole\n"
            "                          training set shares one reliably invokable hook instead of\n"
            "                          relying on each sample's own long natural-language caption.\n"
            "                          Include it in the caption at inference time to invoke the\n"
            "                          adapter's learned concept.\n"
            "  --tag-position <mode>   prepend | append | replace (default: prepend). 'replace'\n"
            "                          drops the per-sample caption entirely (auto-label is\n"
            "                          skipped too) -- the trigger word IS the whole prompt.\n"
            "\n"
            "Options:\n"
            "  --dit-model <name>      DiT GGUF filename (default: first DiT found)\n"
            "  --text-enc-model <name> Text encoder GGUF filename (default: first found)\n"
            "  --vae-model <name>      VAE GGUF filename (default: first found)\n"
            "  --max-duration <F>      Truncate audio to this many seconds (default: 240)\n"
            "  --vae-chunk <N>         Latent frames per VAE tile (default: 1024)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n"
            "  --skip-existing         Skip samples whose output file already exists\n",
            ACE_VERSION, prog);
}

static int run_prepare(const char * prog, int argc, char ** argv) {
    PrepareArgs a;
    for (int i = 0; i < argc; i++) {
        std::string arg  = argv[i];
        auto        need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "[ace-train] FATAL: %s needs a value\n", name);
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "--models") {
            a.models_dir = need("--models");
        } else if (arg == "--dataset") {
            a.dataset_dir = need("--dataset");
        } else if (arg == "--output") {
            a.output_dir = need("--output");
        } else if (arg == "--dit-model") {
            a.dit_model = need("--dit-model");
        } else if (arg == "--text-enc-model") {
            a.text_enc_model = need("--text-enc-model");
        } else if (arg == "--vae-model") {
            a.vae_model = need("--vae-model");
        } else if (arg == "--lm-model") {
            a.lm_model = need("--lm-model");
        } else if (arg == "--max-duration") {
            a.max_duration = (float) atof(need("--max-duration"));
        } else if (arg == "--vae-chunk") {
            a.vae_chunk = atoi(need("--vae-chunk"));
        } else if (arg == "--vae-overlap") {
            a.vae_overlap = atoi(need("--vae-overlap"));
        } else if (arg == "--skip-existing") {
            a.skip_existing = true;
        } else if (arg == "--trigger-word") {
            a.trigger_word = need("--trigger-word");
        } else if (arg == "--tag-position") {
            a.tag_position = need("--tag-position");
        } else if (arg == "-h" || arg == "--help") {
            usage_prepare(prog);
            return 0;
        } else {
            fprintf(stderr, "[ace-train] FATAL: unknown argument '%s'\n", arg.c_str());
            usage_prepare(prog);
            return 1;
        }
    }

    if (a.models_dir.empty() || a.dataset_dir.empty() || a.output_dir.empty()) {
        fprintf(stderr, "[ace-train] FATAL: --models, --dataset and --output are all required\n");
        usage_prepare(prog);
        return 1;
    }
    if (a.tag_position != "prepend" && a.tag_position != "append" && a.tag_position != "replace") {
        fprintf(stderr, "[ace-train] FATAL: --tag-position must be prepend, append or replace (got '%s')\n",
                a.tag_position.c_str());
        return 1;
    }

    ModelRegistry reg;
    if (!registry_scan(&reg, a.models_dir.c_str())) {
        fprintf(stderr, "[ace-train] FATAL: no models found in %s\n", a.models_dir.c_str());
        return 1;
    }

    auto resolve = [&](const std::vector<ModelEntry> & bucket, const std::string & override_name,
                       const char * kind) -> std::string {
        if (!override_name.empty()) {
            const ModelEntry * e = registry_find(bucket, override_name.c_str());
            if (!e) {
                fprintf(stderr, "[ace-train] FATAL: %s model '%s' not found in %s\n", kind, override_name.c_str(),
                        a.models_dir.c_str());
                exit(1);
            }
            return e->path;
        }
        if (bucket.empty()) {
            fprintf(stderr, "[ace-train] FATAL: no %s model found in %s\n", kind, a.models_dir.c_str());
            exit(1);
        }
        return bucket[0].path;
    };

    DiTPrepareParams params;
    params.dit_path      = resolve(reg.dit, a.dit_model, "DiT");
    params.text_enc_path = resolve(reg.text_enc, a.text_enc_model, "Text-Enc");
    params.vae_path       = resolve(reg.vae, a.vae_model, "VAE");
    params.vae_chunk      = a.vae_chunk;
    params.vae_overlap    = a.vae_overlap;
    params.max_duration   = a.max_duration;
    params.custom_tag     = a.trigger_word;
    params.tag_position   = a.tag_position;

    std::string lm_path;
    if (!a.lm_model.empty()) {
        const ModelEntry * e = registry_find(reg.lm, a.lm_model.c_str());
        if (!e) {
            fprintf(stderr, "[ace-train] FATAL: LM model '%s' not found in %s\n", a.lm_model.c_str(),
                    a.models_dir.c_str());
            return 1;
        }
        lm_path = e->path;
    } else if (!reg.lm.empty()) {
        lm_path = reg.lm[0].path;
    }
    fprintf(stderr, "[ace-train] DiT: %s\n", params.dit_path.c_str());
    fprintf(stderr, "[ace-train] Text-Enc: %s\n", params.text_enc_path.c_str());
    fprintf(stderr, "[ace-train] VAE: %s\n", params.vae_path.c_str());
    fprintf(stderr, "[ace-train] Auto-label LM: %s\n", lm_path.empty() ? "(none -- captions must come from sidecars)"
                                                                      : lm_path.c_str());
    if (!a.trigger_word.empty()) {
        fprintf(stderr, "[ace-train] Trigger word: \"%s\" (%s) -- include this in the caption at inference time\n",
                a.trigger_word.c_str(), a.tag_position.c_str());
    } else {
        fprintf(stderr, "[ace-train] Trigger word: (none -- each sample's own caption used verbatim)\n");
    }

    std::vector<DiTPrepareLabel> labels = dit_prepare_scan_dataset(a.dataset_dir);
    if (labels.empty()) {
        fprintf(stderr, "[ace-train] FATAL: no .wav/.mp3 files found in %s\n", a.dataset_dir.c_str());
        return 1;
    }
    fprintf(stderr, "[ace-train] Found %zu audio files in %s\n", labels.size(), a.dataset_dir.c_str());

    std::error_code ec;
    std::filesystem::create_directories(a.output_dir, ec);

    // EVICT_NEVER: VAE-Enc, Text-Enc and Cond-Enc are all small and used on
    // every sample. Under EVICT_STRICT each require() would evict the other
    // two, forcing a full reload per sample per module -- pure overhead for
    // a batch tool with no VRAM pressure between these three.
    ModelStore * store = store_create(EVICT_NEVER);

    int n_ok = 0, n_skip = 0, n_fail = 0;
    for (size_t i = 0; i < labels.size(); i++) {
        DiTPrepareLabel & label       = labels[i];
        std::string        out_path   = a.output_dir + "/" + label.stem + ".gguf";

        if (a.skip_existing && registry_is_file(out_path.c_str())) {
            fprintf(stderr, "[ace-train] [%zu/%zu] Skipping %s (exists)\n", i + 1, labels.size(), label.stem.c_str());
            n_skip++;
            continue;
        }

        fprintf(stderr, "[ace-train] [%zu/%zu] Encoding %s\n", i + 1, labels.size(), label.stem.c_str());

        DiTTrainSample sample;
        if (!dit_prepare_encode_sample(store, params, label, lm_path.empty() ? nullptr : lm_path.c_str(), &sample)) {
            fprintf(stderr, "[ace-train] FAILED: %s\n", label.stem.c_str());
            n_fail++;
            continue;
        }

        if (!dit_train_sample_write(sample, out_path)) {
            fprintf(stderr, "[ace-train] FAILED to write: %s\n", out_path.c_str());
            n_fail++;
            continue;
        }
        n_ok++;
    }

    store_free(store);

    fprintf(stderr, "[ace-train] Done: %d encoded, %d skipped, %d failed (of %zu)\n", n_ok, n_skip, n_fail,
            labels.size());
    return (n_fail > 0 && n_ok == 0) ? 1 : 0;
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
        return run_prepare(argv[0], argc - 2, argv + 2);
    }
    fprintf(stderr, "[ace-train] FATAL: unknown subcommand '%s' (expected 'fit' or 'prepare')\n", cmd.c_str());
    return 1;
}
