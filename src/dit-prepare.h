#pragma once
// dit-prepare.h: dataset preparation for LoRA training (Phase 4, see
// ../TRAINING_DEV.md). Scans a directory of audio files + sidecar labels,
// auto-labels missing captions via the understand pipeline, and encodes
// each sample into the tensor cache format src/dit-train-data.h reads.
//
// Sidecar schema (matches acestep-repo's dataset_builder_modules, i.e. the
// Python LoRA trainer this port targets):
//   {name}.json         caption, bpm, keyscale, timesignature, language,
//                       lyrics, is_instrumental (any subset; all optional)
//   {name}.lyrics.txt   raw lyrics text (used when the JSON has none)
//   {name}.txt          same, alternate name
//   {name}.caption.txt  raw caption text (used when the JSON has none)
//
// Encoding matches ace-synth's own inference-time conditioning exactly
// (text prompt via src/dit-prompt.h, lyrics via the same qwen3_embed_lookup
// path, CondEncoder timbre placeholder via the DiT's own silence_latent),
// NOT the Python LoRA trainer's dataset_builder_modules/preprocess_lyrics.py
// (which tokenizes raw unwrapped lyrics) or its zeros(1,1,64) timbre dummy.
// That choice is deliberate: a LoRA trained here is used via ace-synth, so
// it must see the exact conditioning distribution ace-synth actually
// produces. See TRAINING_DEV.md's Phase 4 decision record.

#include "audio-io.h"
#include "bpe.h"
#include "cond-enc.h"
#include "dit-prompt.h"
#include "dit-train-data.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-understand.h"
#include "qwen3-enc.h"
#include "task-types.h"
#include "vae-enc.h"
#include "yyjson.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct DiTPrepareLabel {
    std::string audio_path;
    std::string stem;  // filename without extension, used for output naming
    std::string caption;
    std::string lyrics;
    int         bpm            = 0;
    std::string keyscale;
    std::string timesignature;
    std::string language;  // "" = unknown
    bool        is_instrumental = true;
};

static std::string dit_prepare_read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return "";
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) {
        fclose(f);
        return "";
    }
    std::string buf((size_t) len, '\0');
    size_t      nr = fread(&buf[0], 1, (size_t) len, f);
    fclose(f);
    buf.resize(nr);
    // trim trailing whitespace/newlines (plain-text sidecars)
    while (!buf.empty() && (buf.back() == '\n' || buf.back() == '\r' || buf.back() == ' ')) {
        buf.pop_back();
    }
    return buf;
}

// Parse the JSON sidecar. Missing fields are left at the label's current
// value (caller pre-fills defaults), matching the dataset builder's "fill
// what's provided, leave the rest to auto-label" behavior.
static void dit_prepare_parse_json_sidecar(const std::string & path, DiTPrepareLabel * label) {
    std::string text = dit_prepare_read_file(path);
    if (text.empty()) {
        return;
    }
    yyjson_doc * doc = yyjson_read(text.c_str(), text.size(), 0);
    if (!doc) {
        fprintf(stderr, "[Prepare] WARNING: malformed JSON sidecar %s\n", path.c_str());
        return;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * v;
    if ((v = yyjson_obj_get(root, "caption")) && yyjson_is_str(v)) {
        label->caption = std::string(yyjson_get_str(v), yyjson_get_len(v));
    }
    if ((v = yyjson_obj_get(root, "lyrics")) && yyjson_is_str(v)) {
        label->lyrics = std::string(yyjson_get_str(v), yyjson_get_len(v));
    }
    if ((v = yyjson_obj_get(root, "bpm")) && yyjson_is_int(v)) {
        label->bpm = (int) yyjson_get_int(v);
    }
    if ((v = yyjson_obj_get(root, "keyscale")) && yyjson_is_str(v)) {
        label->keyscale = std::string(yyjson_get_str(v), yyjson_get_len(v));
    }
    if ((v = yyjson_obj_get(root, "timesignature")) && yyjson_is_str(v)) {
        label->timesignature = std::string(yyjson_get_str(v), yyjson_get_len(v));
    }
    if ((v = yyjson_obj_get(root, "language")) && yyjson_is_str(v)) {
        std::string lang = std::string(yyjson_get_str(v), yyjson_get_len(v));
        if (lang != "unknown") {
            label->language = lang;
        }
    }
    if ((v = yyjson_obj_get(root, "is_instrumental")) && yyjson_is_bool(v)) {
        label->is_instrumental = yyjson_get_bool(v);
    }
    yyjson_doc_free(doc);
}

// Scan a dataset directory for audio files + sidecars. Only .wav/.mp3 are
// decodable today (src/audio-io.h); .flac/.ogg/.opus sidecars are reported
// and skipped rather than silently ignored.
static std::vector<DiTPrepareLabel> dit_prepare_scan_dataset(const std::string & dataset_dir) {
    std::vector<std::string> files;
    registry_list_dir(dataset_dir.c_str(), &files);
    std::sort(files.begin(), files.end());

    std::vector<DiTPrepareLabel> labels;
    for (const auto & fname : files) {
        std::string ext;
        size_t      dot = fname.rfind('.');
        if (dot != std::string::npos) {
            ext = fname.substr(dot);
        }
        for (char & c : ext) {
            c = (char) tolower((unsigned char) c);
        }

        bool is_wav_mp3 = (ext == ".wav" || ext == ".mp3");
        bool is_unsupported_audio = (ext == ".flac" || ext == ".ogg" || ext == ".opus");
        if (is_unsupported_audio) {
            fprintf(stderr,
                    "[Prepare] WARNING: %s has an unsupported audio format for this build (.flac/.ogg/.opus "
                    "decoding is not implemented in src/audio-io.h), skipping\n",
                    fname.c_str());
            continue;
        }
        if (!is_wav_mp3) {
            continue;
        }

        DiTPrepareLabel label;
        label.stem       = fname.substr(0, dot);
        label.audio_path = dataset_dir + "/" + fname;
        label.language   = "";

        std::string base = dataset_dir + "/" + label.stem;
        dit_prepare_parse_json_sidecar(base + ".json", &label);

        if (label.lyrics.empty()) {
            std::string lyrics_path = base + ".lyrics.txt";
            if (!registry_is_file(lyrics_path.c_str())) {
                lyrics_path = base + ".txt";
            }
            if (registry_is_file(lyrics_path.c_str())) {
                label.lyrics = dit_prepare_read_file(lyrics_path);
            }
        }
        if (label.caption.empty()) {
            std::string caption_path = base + ".caption.txt";
            if (registry_is_file(caption_path.c_str())) {
                label.caption = dit_prepare_read_file(caption_path);
            }
        }
        if (!label.lyrics.empty()) {
            label.is_instrumental = false;
        }

        labels.push_back(std::move(label));
    }
    return labels;
}

struct DiTPrepareParams {
    std::string dit_path;
    std::string text_enc_path;
    std::string vae_path;
    int         vae_chunk    = 1024;
    int         vae_overlap  = 64;
    float       max_duration = 240.0f;  // seconds; matches the Python trainer's default
};

// Fill caption/lyrics/bpm/keyscale/language via the understand pipeline when
// the sidecars left the caption empty ("Auto Label Data" in the Python
// trainer's UI). understand's own audio-derived duration is ignored: the
// metas string always uses the real measured sample length (see
// dit_prepare_encode_sample), matching how the dataset builder treats
// duration as an auto-read property of the file, not an LM guess.
static bool dit_prepare_auto_label(ModelStore *              store,
                                   const char *              lm_path,
                                   const DiTPrepareParams & params,
                                   const float *             interleaved_audio,
                                   int                       T_audio,
                                   DiTPrepareLabel *         label) {
    AceUnderstandParams up;
    ace_understand_default_params(&up);
    up.model_path = lm_path;
    up.dit_path   = params.dit_path.c_str();
    up.vae_path   = params.vae_path.c_str();
    up.vae_chunk  = params.vae_chunk;
    up.vae_overlap = params.vae_overlap;

    AceUnderstand * u = ace_understand_load(store, &up);
    if (!u) {
        fprintf(stderr, "[Prepare] WARNING: auto-label unavailable (understand pipeline failed to init)\n");
        return false;
    }

    AceRequest req;
    request_init(&req);
    AceRequest out;
    request_init(&out);

    int rc = ace_understand_generate(u, interleaved_audio, T_audio, nullptr, 0, &req, &out);
    ace_understand_free(u);
    if (rc != 0) {
        fprintf(stderr, "[Prepare] WARNING: auto-label failed for this sample\n");
        return false;
    }

    if (label->caption.empty()) {
        label->caption = out.caption;
    }
    if (label->lyrics.empty() && !out.lyrics.empty() && out.lyrics != "[Instrumental]") {
        label->lyrics          = out.lyrics;
        label->is_instrumental = false;
    }
    if (label->bpm <= 0) {
        label->bpm = out.bpm;
    }
    if (label->keyscale.empty()) {
        label->keyscale = out.keyscale;
    }
    if (label->timesignature.empty()) {
        label->timesignature = out.timesignature;
    }
    if (label->language.empty() && out.vocal_language != "unknown") {
        label->language = out.vocal_language;
    }
    fprintf(stderr, "[Prepare] Auto-labeled %s: caption=%zu chars, lyrics=%zu chars, bpm=%d\n", label->stem.c_str(),
            label->caption.size(), label->lyrics.size(), label->bpm);
    return true;
}

// Encode one labeled sample into a DiTTrainSample, ready for
// dit_train_sample_write(). Returns false on any I/O or model error.
static bool dit_prepare_encode_sample(ModelStore *              store,
                                      const DiTPrepareParams & params,
                                      DiTPrepareLabel &         label,
                                      const char *              lm_path_or_null,
                                      DiTTrainSample *          out) {
    // ---- 1. Load audio, resample to 48kHz stereo, truncate to max_duration ----
    int     T_planar = 0;
    float * planar    = audio_read_48k(label.audio_path.c_str(), &T_planar);
    if (!planar) {
        fprintf(stderr, "[Prepare] FATAL: failed to load %s\n", label.audio_path.c_str());
        return false;
    }
    int max_samples = (int) (params.max_duration * 48000.0f);
    int T_audio     = (T_planar > max_samples) ? max_samples : T_planar;

    float * interleaved = audio_planar_to_interleaved(planar, T_audio);
    free(planar);
    if (!interleaved) {
        fprintf(stderr, "[Prepare] FATAL: OOM converting %s to interleaved\n", label.audio_path.c_str());
        return false;
    }
    float duration_sec = (float) T_audio / 48000.0f;

    // ---- 2. Auto-label if caption is missing -------------------------------
    if (label.caption.empty() && lm_path_or_null) {
        dit_prepare_auto_label(store, lm_path_or_null, params, interleaved, T_audio, &label);
    }
    if (label.caption.empty()) {
        fprintf(stderr, "[Prepare] FATAL: %s has no caption (sidecar missing and no --lm-model for auto-label)\n",
                label.stem.c_str());
        free(interleaved);
        return false;
    }

    // ---- 3. VAE encode: audio -> target_latents [T, 64] --------------------
    ModelKey vae_key = {};
    vae_key.kind     = MODEL_VAE_ENC;
    vae_key.path     = params.vae_path;
    VAEEncoder * vae_enc = store_require_vae_enc(store, vae_key);
    if (!vae_enc) {
        fprintf(stderr, "[Prepare] FATAL: store_require_vae_enc failed\n");
        free(interleaved);
        return false;
    }
    int                 max_T_lat = (T_audio / 1920) + 64;
    std::vector<float> target_latents((size_t) max_T_lat * 64, 0.0f);
    int                 T_latent;
    {
        ModelHandle guard(store, vae_enc);
        T_latent = vae_enc_encode_tiled(vae_enc, interleaved, T_audio, target_latents.data(), max_T_lat,
                                        params.vae_chunk, params.vae_overlap);
    }
    free(interleaved);
    if (T_latent <= 0) {
        fprintf(stderr, "[Prepare] FATAL: VAE encode failed for %s\n", label.audio_path.c_str());
        return false;
    }
    target_latents.resize((size_t) T_latent * 64);

    // The DiT patchifies in groups of patch_size frames (dit_ggml_build_graph
    // reshapes [in_channels, T, N] -> [in_channels*P, T/P, N], which requires
    // T % P == 0). ace-synth rounds T up the same way for --src-audio inputs
    // (src/pipeline-synth-ops.cpp:330); VAE-encoded length has no reason to
    // already be a multiple of P, so pad the tail with silence_latent up to
    // the next multiple, same as inference does for a source shorter than
    // the rounded T. The label for the padded region is silence too, so the
    // flow-matching loss simply asks the model to predict silence there.
    const DiTMeta * meta = store_dit_meta(store, params.dit_path.c_str());
    if (!meta || meta->silence_full.empty()) {
        fprintf(stderr, "[Prepare] FATAL: store_dit_meta failed for %s\n", params.dit_path.c_str());
        return false;
    }
    int patch_size = meta->cfg.patch_size;
    int T_padded   = ((T_latent + patch_size - 1) / patch_size) * patch_size;
    if (T_padded > T_latent) {
        target_latents.resize((size_t) T_padded * 64);
        memcpy(&target_latents[(size_t) T_latent * 64], meta->silence_full.data(),
               (size_t) (T_padded - T_latent) * 64 * sizeof(float));
    }
    T_latent = T_padded;

    if (T_latent > 15000) {
        fprintf(stderr, "[Prepare] FATAL: %s exceeds the 15000-frame silence_latent buffer (T_latent=%d)\n",
                label.stem.c_str(), T_latent);
        return false;
    }

    // ---- 4. Text + lyric prompt, tokenize, encode --------------------------
    std::string lyrics_for_prompt = (!label.is_instrumental && !label.lyrics.empty()) ? label.lyrics : "[Instrumental]";
    std::string text_prompt, lyric_prompt;
    dit_build_prompt_strings(DIT_INSTR_TEXT2MUSIC, label.caption, lyrics_for_prompt, label.bpm, label.keyscale,
                             label.timesignature, label.language, duration_sec, text_prompt, lyric_prompt);

    BPETokenizer * bpe = store_bpe(store, params.text_enc_path.c_str());
    if (!bpe) {
        fprintf(stderr, "[Prepare] FATAL: store_bpe failed\n");
        return false;
    }
    std::vector<int> text_ids  = bpe_encode(bpe, text_prompt, true);
    std::vector<int> lyric_ids = bpe_encode(bpe, lyric_prompt, true);
    int               S_text   = (int) text_ids.size();
    int               S_lyric  = (int) lyric_ids.size();

    ModelKey text_key = {};
    text_key.kind     = MODEL_TEXT_ENC;
    text_key.path     = params.text_enc_path;
    Qwen3GGML * te = store_require_text_enc(store, text_key);
    if (!te) {
        fprintf(stderr, "[Prepare] FATAL: store_require_text_enc failed\n");
        return false;
    }
    int                 H_text = te->cfg.hidden_size;
    std::vector<float> text_hidden((size_t) H_text * S_text);
    std::vector<float> lyric_embed((size_t) H_text * S_lyric);
    {
        ModelHandle guard(store, te);
        qwen3_forward(te, text_ids.data(), S_text, text_hidden.data());
        qwen3_embed_lookup(te, lyric_ids.data(), S_lyric, lyric_embed.data());
    }

    // ---- 5. CondEncoder: (text, lyric, timbre=silence placeholder) --------
    // 1-frame silence timbre placeholder, matching ace-synth's own
    // no-ref-audio inference convention (see file header for why this
    // differs from the Python trainer's zeros(1,1,64) dummy).
    ModelKey cond_key = {};
    cond_key.kind     = MODEL_COND_ENC;
    cond_key.path     = params.dit_path;
    CondGGML * ce = store_require_cond_enc(store, cond_key);
    if (!ce) {
        fprintf(stderr, "[Prepare] FATAL: store_require_cond_enc failed\n");
        return false;
    }
    std::vector<float> encoder_hidden;
    int                 enc_S = 0;
    {
        ModelHandle guard(store, ce);
        cond_ggml_forward(ce, text_hidden.data(), S_text, lyric_embed.data(), S_lyric, meta->silence_full.data(), 1,
                          encoder_hidden, &enc_S);
    }

    // ---- 6. context_latents: silence[:T] as src + ones as mask (text2music) ----
    std::vector<float> context_latents((size_t) T_latent * 128);
    for (int t = 0; t < T_latent; t++) {
        memcpy(&context_latents[(size_t) t * 128], &meta->silence_full[(size_t) t * 64], 64 * sizeof(float));
        for (int c = 0; c < 64; c++) {
            context_latents[(size_t) t * 128 + 64 + c] = 1.0f;
        }
    }

    out->T              = T_latent;
    out->enc_S          = enc_S;
    out->target_latents = std::move(target_latents);
    out->context_latents = std::move(context_latents);
    out->encoder_hidden = std::move(encoder_hidden);
    out->caption         = label.caption;
    return true;
}
