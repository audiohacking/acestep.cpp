#pragma once
// dit-train-data.h: per-sample training tensor cache, read + write.
//
// A "sample" is exactly the tensors src/dit-train.h needs for one training
// step, unpadded (each sample keeps its own T and enc_S, no batching):
//
//   target_latents     [64, T]      VAE-encoded audio (x0)
//   context_latents    [128, T]     src||mask context (silence+ones for text2music)
//   encoder_hidden     [H_enc, enc_S]  condition encoder output
//
// No attention_mask / encoder_attention_mask tensors: since every sample is
// stored and trained at its own exact (unpadded) length, an all-valid mask
// is already correct (src/dit-train.h builds one internally). Padding only
// becomes necessary if/when samples are ever batched to N>1 at fit time.
//
// Stored as GGUF (one file per sample) purely as a tensor container: no
// model semantics, just three named tensors + a caption string for
// human-readable dataset browsing. Written with gguf_add_tensor +
// gguf_write_to_file(only_meta=false) (single-pass write, tensors carry
// their own ->data). Read via the existing gf_load() mmap + KV/tensor
// accessors from gguf-weights.h, so no second GGUF parser is needed.

#include "gguf-weights.h"
#include "model-registry.h"  // registry_list_dir

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

struct DiTTrainSample {
    int                 T     = 0;
    int                 enc_S = 0;
    std::vector<float> target_latents;   // [64, T] flat, ggml ne=(64,T) row order
    std::vector<float> context_latents;  // [128, T] flat
    std::vector<float> encoder_hidden;   // [H_enc, enc_S] flat
    std::string         caption;          // optional, for logging only
};

// Write one sample to a GGUF file.
static bool dit_train_sample_write(const DiTTrainSample & s, const std::string & path) {
    if (s.T <= 0 || s.enc_S <= 0) {
        fprintf(stderr, "[TrainData] FATAL: invalid sample shape T=%d enc_S=%d\n", s.T, s.enc_S);
        return false;
    }
    int64_t H_enc = (int64_t) s.encoder_hidden.size() / s.enc_S;
    if (H_enc * s.enc_S != (int64_t) s.encoder_hidden.size()) {
        fprintf(stderr, "[TrainData] FATAL: encoder_hidden size not divisible by enc_S\n");
        return false;
    }

    size_t data_bytes = ((size_t) 64 * s.T + (size_t) 128 * s.T + (size_t) H_enc * s.enc_S) * sizeof(float);
    struct ggml_init_params params = { ggml_tensor_overhead() * 8 + data_bytes + 4096, NULL, /*no_alloc=*/false };
    struct ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        return false;
    }

    struct ggml_tensor * t_target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, s.T);
    ggml_set_name(t_target, "target_latents");
    memcpy(t_target->data, s.target_latents.data(), ggml_nbytes(t_target));

    struct ggml_tensor * t_ctx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, s.T);
    ggml_set_name(t_ctx, "context_latents");
    memcpy(t_ctx->data, s.context_latents.data(), ggml_nbytes(t_ctx));

    struct ggml_tensor * t_enc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_enc, s.enc_S);
    ggml_set_name(t_enc, "encoder_hidden");
    memcpy(t_enc->data, s.encoder_hidden.data(), ggml_nbytes(t_enc));

    struct gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "acestep.tensors.caption", s.caption.c_str());
    gguf_add_tensor(gguf, t_target);
    gguf_add_tensor(gguf, t_ctx);
    gguf_add_tensor(gguf, t_enc);

    bool ok = gguf_write_to_file(gguf, path.c_str(), /*only_meta=*/false);
    if (!ok) {
        fprintf(stderr, "[TrainData] FATAL: failed to write %s\n", path.c_str());
    }

    gguf_free(gguf);
    ggml_free(ctx);
    return ok;
}

// Load one sample from a GGUF file (mmap, copy out, close -- no dangling
// mapping kept open past this call).
static bool dit_train_sample_load(DiTTrainSample * s, const std::string & path) {
    GGUFModel gf;
    if (!gf_load(&gf, path.c_str())) {
        return false;
    }

    auto get = [&](const char * name, std::vector<float> * out, int * ne0, int * ne1) -> bool {
        int64_t idx = gguf_find_tensor(gf.gguf, name);
        if (idx < 0) {
            fprintf(stderr, "[TrainData] FATAL: %s missing tensor '%s'\n", path.c_str(), name);
            return false;
        }
        struct ggml_tensor * t = ggml_get_tensor(gf.meta, name);
        int64_t               n = ggml_nelements(t);
        out->resize((size_t) n);
        memcpy(out->data(), gf_get_data(gf, name), (size_t) n * sizeof(float));
        *ne0 = (int) t->ne[0];
        *ne1 = (int) t->ne[1];
        return true;
    };

    int target_ne0, target_ne1, ctx_ne0, ctx_ne1, enc_ne0, enc_ne1;
    bool ok = get("target_latents", &s->target_latents, &target_ne0, &target_ne1) &&
             get("context_latents", &s->context_latents, &ctx_ne0, &ctx_ne1) &&
             get("encoder_hidden", &s->encoder_hidden, &enc_ne0, &enc_ne1);

    if (ok) {
        if (target_ne1 != ctx_ne1) {
            fprintf(stderr, "[TrainData] FATAL: %s target_latents T=%d != context_latents T=%d\n", path.c_str(),
                    target_ne1, ctx_ne1);
            ok = false;
        }
        s->T     = target_ne1;
        s->enc_S = enc_ne1;

        int64_t meta_idx = gguf_find_key(gf.gguf, "acestep.tensors.caption");
        s->caption       = meta_idx >= 0 ? gguf_get_val_str(gf.gguf, meta_idx) : "";
    }

    gf_close(&gf);
    return ok;
}

// List every *.gguf file in a directory (sample tensor cache), sorted.
static std::vector<std::string> dit_train_scan_tensor_dir(const std::string & dir) {
    std::vector<std::string> files;
    registry_list_dir(dir.c_str(), &files);
    std::sort(files.begin(), files.end());

    std::vector<std::string> paths;
    for (const auto & f : files) {
        if (f.size() > 5 && f.compare(f.size() - 5, 5, ".gguf") == 0) {
            paths.push_back(dir + "/" + f);
        }
    }
    return paths;
}
