#pragma once
// audio-analysis.h: DSP-based BPM/key detection via acebeat (vendor/acebeat/,
// MIT-licensed, first-party), used to complement -- and correct -- the LM's
// own numeric-metadata guesses in ace-understand and ace-train prepare's
// auto-label.
//
// Why: the LM (a generative captioning model) is unreliable at precise
// numeric musical attributes. Measured on a real track with known ground
// truth (bpm=126, key=F minor): the LM guessed bpm=71, key=C# minor, while
// DSP analysis matched the true bpm exactly. See TRAINING_DEV.md for the
// full writeup -- a wrong bpm/key baked into training conditioning (vs. a
// different guess at generation time) was the root cause of a "trained
// LoRA sounds nothing like the source" bug.
//
// Previously used vendored Essentia (AGPL-3.0); replaced with acebeat, an
// independent, from-scratch, MIT-licensed reimplementation built
// specifically to remove that AGPL liability from this MIT-licensed
// project. See vendor/acebeat/README.md for its clean-room statement and
// TRAINING_DEV.md for the real-corpus accuracy gate acebeat was validated
// against before this swap (bpm: 84.7% exact-or-octave-tolerant vs.
// Essentia's own output; key: 58.2% exact / ~71% combined across ~1900
// real tracks).
//
// audio_analyze_bpm_key_from_file() (not the raw-buffer function below) is
// the entry point real callers should use: it decodes the file at its
// native rate and resamples directly to 44100 Hz in one hop -- the same
// pipeline shape the accuracy numbers above were measured with. Analyzing
// a buffer already resampled to 48kHz by the rest of the pipeline (a
// second resample 48k->44.1k on top of that) measurably hurt the old
// Essentia backend (126.0 bpm exact vs. 144.6 bpm on a confirmed-bpm=126
// track); there's no reason to reintroduce that extra hop here.

#include "audio-io.h"
#include "audio-resample.h"

#include "acebeat/acebeat.h"

#include <cstdlib>
#include <string>
#include <vector>

struct AudioAnalysisResult {
    float       bpm             = 0.0f;
    float       bpm_confidence  = 0.0f;
    std::string key;
    std::string scale;
    float       key_strength    = 0.0f;
};

static const char * AUDIO_ANALYSIS_PITCH_NAMES[12] = { "C", "C#", "D", "D#", "E", "F",
                                                         "F#", "G", "G#", "A", "A#", "B" };

// Low-level: PLANAR stereo float samples ([L: T samples][R: T samples],
// matching audio_read()/audio_resample()'s own layout -- NOT interleaved),
// T frames, at sample_rate Hz. Prefer audio_analyze_bpm_key_from_file()
// below. Returns false on internal error.
static bool audio_analyze_bpm_key_buf(const float * planar, int T, int sample_rate, AudioAnalysisResult * out) {
    if (!planar || T <= 0 || !out) {
        return false;
    }

    int     T44 = 0;
    float * resampled = (sample_rate == 44100) ? nullptr : audio_resample(planar, T, sample_rate, 44100, 2, &T44);
    const float * src = resampled ? resampled : planar;
    if (!resampled) {
        T44 = T;
    }

    std::vector<float> mono((size_t) T44);
    for (int i = 0; i < T44; i++) {
        mono[(size_t) i] = 0.5f * (src[(size_t) i] + src[(size_t) (T44 + i)]);  // planar downmix
    }
    if (resampled) {
        free(resampled);
    }

    acebeat::Result r;
    if (!acebeat::analyze_mono(mono.data(), (int) mono.size(), 44100, &r)) {
        return false;
    }

    out->bpm            = r.bpm;
    out->bpm_confidence = r.bpm_confidence;
    out->key = (r.key_pitch_class >= 0 && r.key_pitch_class < 12) ? AUDIO_ANALYSIS_PITCH_NAMES[r.key_pitch_class] : "";
    out->scale        = r.key_is_minor ? "minor" : "major";
    out->key_strength = r.key_strength;
    return true;
}

// Decodes path at its native sample rate and resamples directly to 44100 Hz
// (a no-op if already 44100 -- the common case for mp3 sources). This is
// the entry point real callers should use. Returns false if the file can't
// be read.
static bool audio_analyze_bpm_key_from_file(const char * path, AudioAnalysisResult * out) {
    int     T_native  = 0;
    int     sr_native = 0;
    float * raw        = audio_read(path, &T_native, &sr_native);
    if (!raw) {
        return false;
    }
    bool ok = audio_analyze_bpm_key_buf(raw, T_native, sr_native, out);
    free(raw);
    return ok;
}
