#pragma once
// chroma.h: 12-bin pitch-class ("chroma") profile extraction from a mono
// signal, for use by key.h. Standard technique: magnitude-weighted
// accumulation of spectral energy into pitch classes via the equal-
// temperament frequency-to-MIDI mapping, averaged over non-silent frames.
// Written from the general published description of chroma features, not
// derived from any specific third-party implementation.

#include <vector>

namespace acebeat {

struct ChromaParams {
    // 8192 samples @ 44100 Hz gives ~5.4 Hz bins. Semitone spacing is
    // f*(2^(1/12)-1); below ~90 Hz (F#2) that's narrower than one bin, so
    // neighboring semitones become indistinguishable and get smeared into
    // the wrong pitch class -- a real problem for bass-driven genres,
    // where the bass/kick fundamental is often the single strongest tonal
    // cue. Chroma only needs frequency resolution, not time resolution
    // (the profile is averaged over the whole track anyway), so a much
    // larger window than onset's is the right tradeoff here.
    int   frame_size        = 8192;
    int   hop_size          = 4096;
    float min_freq          = 65.4f;   // C2: below this even an 8192-sample window can't resolve semitones cleanly
    float max_freq          = 5000.0f; // above this, spectral content is mostly harmonics/noise, not fundamental pitch
    float silence_threshold = 1e-6f;   // mean-squared frame energy below this is treated as silence and skipped
};

// Returns a 12-element pitch-class profile (index 0=C, 1=C#, ..., 11=B),
// normalized to sum to 1 (all-zero if the signal is entirely silent).
std::vector<float> compute_chroma(const float * mono, int T, int sample_rate, const ChromaParams & params = {});

}  // namespace acebeat
