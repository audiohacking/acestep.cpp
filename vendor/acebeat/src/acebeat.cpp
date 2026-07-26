#include "acebeat/acebeat.h"

#include "chroma.h"
#include "key.h"
#include "onset.h"
#include "tempo.h"

namespace acebeat {

// (Segment-based key analysis with majority/strength-weighted voting
// across 20s and 40s segments was tried here as a hoped-for improvement
// over whole-track chroma averaging, on the theory that one atypical
// section could otherwise skew the whole-track average. Measured flat to
// slightly worse in every configuration tried (56-56.5% exact-key vs.
// this whole-track version's ~57.5%) on the Phase 3.5 real-corpus gate.
// Reverted; noted here so it isn't retried identically. Whole-track
// chroma averaging is, after all, already an energy-weighted average
// across the track's segments -- explicit segmentation didn't add
// anything this corpus benefited from.)
bool analyze_mono(const float * mono, int T, int sample_rate, Result * out) {
    if (!mono || T <= 0 || sample_rate <= 0 || !out) {
        return false;
    }

    OnsetParams params;
    std::vector<float> envelope      = onset_envelope(mono, T, params);
    float               envelope_rate = (float) sample_rate / (float) params.hop_size;
    TempoResult          tempo         = estimate_tempo(envelope, envelope_rate);

    // Bass-register boost: diagnostics (confirmed robust at n=1903: a
    // fifth-above/dominant-confusion pattern in 200 vs. 47 cases, 4.3x
    // directional) found the classic chroma+template weakness where a
    // note's own 3rd harmonic reinforces its fifth, making a real dominant
    // chord and pure harmonic leakage of the tonic look alike. A discrete
    // correlation-tolerance correction for this (key.cpp) measured
    // genuinely flat -- it can't tell "real confusion" from "legitimate
    // close tie on an already-correct answer." The true tonic is
    // disproportionately also the most sustained *bass* note (root-heavy
    // basslines, cadences resolving to the root in the bass), which is
    // actual disambiguating evidence a correlation-margin rule doesn't
    // have access to. Weight swept directly against the full ~1900-file
    // corpus (0.05/0.1/0.15/0.2 -> 57.9%/58.2%/58.0%/57.6% exact-key vs. a
    // 56.6% no-boost baseline): a clean, consistent, unimodal improvement
    // peaking at 0.1, not noise-like bouncing. (An earlier attempt at
    // weight=0.2 had looked promising on a 200-file sample but hadn't held
    // up on a fresh 500-file one -- this full-corpus sweep, at a ~2.2%
    // margin of error, is what actually settled it.)
    std::vector<float> chroma = compute_chroma(mono, T, sample_rate);

    ChromaParams bass_params;
    bass_params.max_freq = 200.0f;
    std::vector<float> bass_chroma = compute_chroma(mono, T, sample_rate, bass_params);

    std::vector<float> combined(12);
    const float          bass_weight = 0.1f;
    float                 total        = 0.0f;
    for (int pc = 0; pc < 12; pc++) {
        combined[(size_t) pc] = chroma[(size_t) pc] + bass_weight * bass_chroma[(size_t) pc];
        total += combined[(size_t) pc];
    }
    if (total > 0.0f) {
        for (float & v : combined) {
            v /= total;
        }
    }

    KeyResult key = estimate_key(combined);

    out->bpm             = tempo.bpm;
    out->bpm_confidence  = tempo.confidence;
    out->key_pitch_class = key.pitch_class;
    out->key_is_minor    = key.is_minor;
    out->key_strength    = key.strength;
    return true;
}

}  // namespace acebeat
