#pragma once
// audio-analysis.h: DSP-based BPM/key detection via vendored Essentia
// (RhythmExtractor2013, KeyExtractor), used to complement -- and correct --
// the LM's own numeric-metadata guesses in ace-understand and ace-train
// prepare's auto-label.
//
// Why: the LM (a generative captioning model) is unreliable at precise
// numeric musical attributes. Measured on a real track with known ground
// truth (bpm=126, key=F minor): the LM guessed bpm=71, key=C# minor, while
// Essentia's DSP analysis (RhythmExtractor2013 + KeyExtractor) matched the
// true values exactly. See TRAINING_DEV.md for the full writeup -- a wrong
// bpm/key baked into training conditioning (vs. a different guess at
// generation time) was the root cause of a "trained LoRA sounds nothing
// like the source" bug.
//
// RhythmExtractor2013 hardcodes an internal assumption of 44100 Hz (its
// own header notes "TODO only 44100 sample rate is supported"), so input
// audio is resampled to 44100 Hz mono before analysis, regardless of the
// caller's native rate.
//
// audio_analyze_bpm_key_from_file() (not the raw-buffer function below) is
// the entry point real callers should use: it decodes the file at its
// native rate and resamples directly to 44100 Hz in one hop. Analyzing the
// buffer already produced by the rest of the pipeline (which resamples to
// 48kHz first) would mean a second resample 48k->44.1k on top of that --
// measured to matter: on a track with confirmed true bpm=126, one hop
// (native 44100 -> passthrough) gave bpm=126.0 exactly, while going through
// an intermediate 48kHz round-trip (native -> 48k -> 44.1k) gave bpm=144.6.
// Two anti-aliasing lowpass passes evidently smear enough transient detail
// to throw off beat tracking, even though both are "correct" resamples in
// isolation.

#include "audio-io.h"
#include "audio-resample.h"

#include <essentia/algorithmfactory.h>
#include <essentia/essentia.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

struct AudioAnalysisResult {
    float       bpm             = 0.0f;
    float       bpm_confidence  = 0.0f;
    std::string key;
    std::string scale;
    float       key_strength    = 0.0f;
};

// essentia::init() sets up global algorithm registration; safe to call
// exactly once per process regardless of how many times analysis runs.
static void audio_analysis_ensure_init() {
    static std::once_flag flag;
    std::call_once(flag, [] { essentia::init(); });
}

// Low-level: interleaved stereo float samples, T frames, at sample_rate Hz.
// Prefer audio_analyze_bpm_key_from_file() below -- see file header comment
// on why resampling from an already-resampled buffer measurably hurts
// accuracy. Returns false on internal error.
static bool audio_analyze_bpm_key_buf(const float * interleaved, int T, int sample_rate, AudioAnalysisResult * out) {
    if (!interleaved || T <= 0 || !out) {
        return false;
    }
    audio_analysis_ensure_init();

    int     T44 = 0;
    float * resampled =
        (sample_rate == 44100) ? nullptr : audio_resample(interleaved, T, sample_rate, 44100, 2, &T44);
    const float * src = resampled ? resampled : interleaved;
    if (!resampled) {
        T44 = T;
    }

    std::vector<essentia::Real> mono((size_t) T44);
    for (int i = 0; i < T44; i++) {
        mono[(size_t) i] = 0.5f * (src[(size_t) i * 2] + src[(size_t) i * 2 + 1]);
    }
    if (resampled) {
        free(resampled);
    }

    essentia::standard::AlgorithmFactory & factory = essentia::standard::AlgorithmFactory::instance();

    essentia::standard::Algorithm * rhythm = factory.create("RhythmExtractor2013", "method", "multifeature");
    essentia::Real                  bpm = 0, confidence = 0;
    std::vector<essentia::Real>     ticks, estimates, bpmIntervals;
    rhythm->input("signal").set(mono);
    rhythm->output("bpm").set(bpm);
    rhythm->output("ticks").set(ticks);
    rhythm->output("confidence").set(confidence);
    rhythm->output("estimates").set(estimates);
    rhythm->output("bpmIntervals").set(bpmIntervals);
    rhythm->compute();
    delete rhythm;

    essentia::standard::Algorithm * keyExt = factory.create("KeyExtractor");
    std::string                     key, scale;
    essentia::Real                  strength = 0;
    keyExt->input("audio").set(mono);
    keyExt->output("key").set(key);
    keyExt->output("scale").set(scale);
    keyExt->output("strength").set(strength);
    keyExt->compute();
    delete keyExt;

    out->bpm            = bpm;
    out->bpm_confidence = confidence;
    out->key            = key;
    out->scale           = scale;
    out->key_strength    = strength;
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
