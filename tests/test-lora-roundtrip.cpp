// test-lora-roundtrip.cpp: safetensors writer + adapter-merge LoRA round-trip.
//
// Phase 1 of native LoRA training (see ../TRAINING_DEV.md): de-risk the
// output format before touching the training graph. This test:
//
//   1. Generates synthetic LoRA A/B tensors (small random values) for every
//      self_attn/cross_attn q/k/v/o_proj across all DiT layers.
//   2. Writes them as a PEFT-format adapter directory via safetensors-write.h
//      (adapter_model.safetensors + adapter_config.json), exactly like a
//      real trainer checkpoint would.
//   3. Feeds that directory through the exact same adapter_merge() code path
//      ace-synth uses for --adapters, running on the CPU backend so the
//      test needs no GPU.
//   4. Compares the merged weights against a host-computed reference
//      (dequantize base -> add scaling * B@A -> compare) via cosine
//      similarity, the same tolerance style already used by tests/*.py.
//
// Usage:
//   ./test-lora-roundtrip --models <dir> [--rank N] [--alpha N] [--seed N]

#include "adapter-merge.h"
#include "ggml-cpu.h"
#include "gguf-weights.h"
#include "model-registry.h"
#include "safetensors-write.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct TargetTensor {
    std::string name;  // GGUF name, e.g. "decoder.layers.0.self_attn.q_proj.weight"
    int64_t     in_feat;
    int64_t     out_feat;
};

// Enumerate every self_attn/cross_attn q/k/v/o_proj tensor actually present
// in the GGUF (so the test tracks whatever layer count/config the model has).
static std::vector<TargetTensor> enumerate_targets(const GGUFModel & gf, int n_layers) {
    std::vector<TargetTensor> out;
    static const char *       kinds[] = { "self_attn", "cross_attn" };
    static const char *       projs[] = { "q_proj", "k_proj", "v_proj", "o_proj" };

    for (int l = 0; l < n_layers; l++) {
        for (const char * kind : kinds) {
            for (const char * proj : projs) {
                std::string name = "decoder.layers." + std::to_string(l) + "." + kind + "." + proj + ".weight";
                int64_t     idx  = gguf_find_tensor(gf.gguf, name.c_str());
                if (idx < 0) {
                    continue;
                }
                struct ggml_tensor * t = ggml_get_tensor(gf.meta, name.c_str());
                out.push_back({ name, t->ne[0], t->ne[1] });
            }
        }
    }
    return out;
}

// Dequantize a raw tensor buffer (any ggml type) to a float vector.
static std::vector<float> dequant_to_f32(const void * data, int64_t nel, ggml_type type) {
    std::vector<float> out((size_t) nel);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), data, (size_t) nel * sizeof(float));
        return out;
    }
    const struct ggml_type_traits * tt = ggml_get_type_traits(type);
    if (!tt->to_float) {
        fprintf(stderr, "[Test] FATAL: no to_float for type %d\n", (int) type);
        exit(1);
    }
    tt->to_float(data, out.data(), nel);
    return out;
}

static double cosine_sim(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) {
        return (na == 0.0 && nb == 0.0) ? 1.0 : 0.0;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// BF16 round trip, matching the rounding adapter-merge.h applies to A, B and
// the delta before adding to base (PEFT / LyCORIS both round through bf16).
static float bf16_round(float x) {
    ggml_bf16_t b = ggml_fp32_to_bf16(x);
    return ggml_bf16_to_fp32(b);
}

int main(int argc, char ** argv) {
    std::string models_dir;
    int         rank  = 4;
    int         alpha = 8;
    uint32_t    seed  = 42;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--models" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--rank" && i + 1 < argc) {
            rank = atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            alpha = atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = (uint32_t) atoi(argv[++i]);
        }
    }
    if (models_dir.empty()) {
        fprintf(stderr, "Usage: %s --models <dir> [--rank N] [--alpha N] [--seed N]\n", argv[0]);
        return 1;
    }

    ModelRegistry reg;
    if (!registry_scan(&reg, models_dir.c_str()) || reg.dit.empty()) {
        fprintf(stderr, "[Test] FAIL: no DiT model found in %s\n", models_dir.c_str());
        return 1;
    }
    std::string dit_path = reg.dit[0].path;
    fprintf(stderr, "[Test] Using DiT: %s\n", dit_path.c_str());

    GGUFModel gf;
    if (!gf_load(&gf, dit_path.c_str())) {
        fprintf(stderr, "[Test] FAIL: could not load %s\n", dit_path.c_str());
        return 1;
    }

    // Enough layers to exercise fused-QKV, split-QKV and unfused code paths
    // if the model has them, but capped so the test stays fast.
    uint32_t                  block_count = gf_get_u32(gf, "acestep-dit.block_count");
    int                       n_layers    = (int) block_count;
    std::vector<TargetTensor> targets     = enumerate_targets(gf, n_layers);
    if (targets.empty()) {
        fprintf(stderr, "[Test] FAIL: no self_attn/cross_attn q/k/v/o_proj tensors found\n");
        gf_close(&gf);
        return 1;
    }
    fprintf(stderr, "[Test] Found %zu target tensors across %d layers\n", targets.size(), n_layers);

    // ---- 1. Generate synthetic LoRA A/B for every target -------------------
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist_a(0.0f, 0.01f);
    std::normal_distribution<float> dist_b(0.0f, 0.02f);  // nonzero so the round-trip is non-trivial

    struct LoraPair {
        std::vector<float> a;  // [rank, in_feat] row major
        std::vector<float> b;  // [out_feat, rank] row major
    };
    std::vector<LoraPair> pairs(targets.size());
    for (size_t i = 0; i < targets.size(); i++) {
        LoraPair & p = pairs[i];
        p.a.resize((size_t) rank * targets[i].in_feat);
        p.b.resize((size_t) targets[i].out_feat * rank);
        for (float & v : p.a) {
            v = dist_a(rng);
        }
        for (float & v : p.b) {
            v = dist_b(rng);
        }
    }

    // ---- 2. Write PEFT adapter directory -----------------------------------
    fs::path adapter_dir = fs::temp_directory_path() / "acestep-lora-roundtrip-test";
    std::error_code ec;
    fs::remove_all(adapter_dir, ec);
    fs::create_directories(adapter_dir, ec);

    std::vector<STWriteEntry> entries;
    for (size_t i = 0; i < targets.size(); i++) {
        std::string base = "base_model.model." + targets[i].name.substr(strlen("decoder."));
        base              = base.substr(0, base.size() - strlen(".weight"));  // strip trailing ".weight"

        entries.push_back({ base + ".lora_A.weight",
                            "F32",
                            { rank, targets[i].in_feat },
                            pairs[i].a.data(),
                            pairs[i].a.size() * sizeof(float) });
        entries.push_back({ base + ".lora_B.weight",
                            "F32",
                            { targets[i].out_feat, rank },
                            pairs[i].b.data(),
                            pairs[i].b.size() * sizeof(float) });
    }

    fs::path st_path = adapter_dir / "adapter_model.safetensors";
    if (!st_write(st_path.string(), entries)) {
        fprintf(stderr, "[Test] FAIL: could not write %s\n", st_path.string().c_str());
        return 1;
    }

    fs::path   cfg_path = adapter_dir / "adapter_config.json";
    FILE *     cfg_f    = fopen(cfg_path.string().c_str(), "wb");
    if (!cfg_f) {
        fprintf(stderr, "[Test] FAIL: could not write %s\n", cfg_path.string().c_str());
        return 1;
    }
    fprintf(cfg_f,
            "{\"r\":%d,\"lora_alpha\":%d,\"lora_dropout\":0.1,"
            "\"target_modules\":[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"],\"bias\":\"none\"}\n",
            rank, alpha);
    fclose(cfg_f);
    fprintf(stderr, "[Test] Wrote adapter to %s (%zu tensors)\n", adapter_dir.string().c_str(), entries.size());

    // ---- 3. Run the real adapter_merge() path on CPU -----------------------
    // Mirrors what gguf-weights.h loaders do before wctx_alloc: push one
    // PendingCopy per target tensor, keyed by its raw mmap pointer (the same
    // key adapter_merge_lora looks up via pending_idx).
    WeightCtx wctx;
    wctx_init(&wctx, (int) targets.size());
    std::vector<const void *> base_ptrs(targets.size());
    for (size_t i = 0; i < targets.size(); i++) {
        base_ptrs[i] = gf_get_data(gf, targets[i].name.c_str());
        struct ggml_tensor * tmeta = ggml_get_tensor(gf.meta, targets[i].name.c_str());
        size_t                nb   = ggml_row_size(tmeta->type, tmeta->ne[0]) * (size_t) tmeta->ne[1];
        wctx.pending.push_back({ nullptr, base_ptrs[i], nb, 0 });
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "[Test] FAIL: could not init CPU backend\n");
        return 1;
    }

    if (!adapter_merge(&wctx, gf, adapter_dir.string().c_str(), /*scale=*/1.0f, backend)) {
        fprintf(stderr, "[Test] FAIL: adapter_merge() reported failure\n");
        return 1;
    }

    // ---- 4. Verify each merged tensor against a host-computed reference ----
    int    n_checked = 0;
    int    n_failed  = 0;
    double min_cossim = 1.0;

    for (size_t i = 0; i < targets.size(); i++) {
        const TargetTensor & tgt   = targets[i];
        struct ggml_tensor *  tmeta = ggml_get_tensor(gf.meta, tgt.name.c_str());
        ggml_type             type  = tmeta->type;
        int64_t               nel   = tgt.in_feat * tgt.out_feat;

        // reference: base_f32 + (alpha/rank) * bf16_round(B @ A), bf16-rounded delta
        std::vector<float> base_f32 = dequant_to_f32(base_ptrs[i], nel, type);

        std::vector<float> a_br(pairs[i].a.size());
        for (size_t k = 0; k < a_br.size(); k++) {
            a_br[k] = bf16_round(pairs[i].a[k]);
        }
        std::vector<float> b_br(pairs[i].b.size());
        for (size_t k = 0; k < b_br.size(); k++) {
            b_br[k] = bf16_round(pairs[i].b[k]);
        }

        float               scaling = (float) alpha / (float) rank;
        std::vector<float>  expected(base_f32.size());
        for (int64_t out_i = 0; out_i < tgt.out_feat; out_i++) {
            for (int64_t in_i = 0; in_i < tgt.in_feat; in_i++) {
                float acc = 0.0f;
                for (int r = 0; r < rank; r++) {
                    acc += b_br[(size_t) out_i * rank + r] * a_br[(size_t) r * tgt.in_feat + in_i];
                }
                float delta                              = bf16_round(scaling * acc);
                expected[(size_t) out_i * tgt.in_feat + in_i] = base_f32[(size_t) out_i * tgt.in_feat + in_i] + delta;
            }
        }

        // actual: whatever adapter_merge staged for this tensor, dequantized
        const WeightCtx::PendingCopy & pc = wctx.pending[i];
        std::vector<float>             actual = dequant_to_f32(pc.src, nel, type);

        double cs = cosine_sim(expected, actual);
        min_cossim = std::min(min_cossim, cs);
        n_checked++;
        bool ok = cs > 0.999;
        if (!ok) {
            n_failed++;
            fprintf(stderr, "[Test] FAIL %s: cossim=%.6f\n", tgt.name.c_str(), cs);
        }
    }

    fprintf(stderr, "[Test] Checked %d tensors, %d failed, min cossim=%.6f\n", n_checked, n_failed, min_cossim);

    ggml_backend_free(backend);
    gf_close(&gf);

    if (n_failed > 0) {
        fprintf(stderr, "[Test] RESULT: FAIL\n");
        return 1;
    }
    fprintf(stderr, "[Test] RESULT: PASS\n");
    return 0;
}
