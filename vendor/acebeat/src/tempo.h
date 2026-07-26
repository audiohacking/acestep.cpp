#pragma once
// tempo.h: autocorrelation-based tempo estimation from an onset-strength
// envelope (see onset.h). Written from the general published description
// of autocorrelation/periodicity-based tempo estimation (see e.g. Ellis,
// "Beat Tracking by Dynamic Programming," 2007, and the broader tempo
// induction literature it surveys) -- a decades-old, widely
// independently-reimplemented technique family, not derived from any
// specific third-party implementation.

#include <vector>

namespace acebeat {

struct TempoResult {
    float bpm        = 0.0f;
    float confidence = 0.0f;  // 0..1
};

// envelope_rate: sample rate of `envelope` itself, i.e.
// (mono sample rate) / (OnsetParams::hop_size), NOT the original audio's
// sample rate.
TempoResult estimate_tempo(const std::vector<float> & envelope, float envelope_rate, float min_bpm = 40.0f,
                           float max_bpm = 208.0f);

}  // namespace acebeat
