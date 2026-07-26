#pragma once
// acebeat.h: public API. See ../README.md for scope and clean-room statement.
//
// acebeat does not decode audio files -- callers provide a mono float
// buffer at a known sample rate. No channel-layout ambiguity: "mono" means
// exactly one sample per frame, nothing else.

namespace acebeat {

struct Result {
    float bpm             = 0.0f;  // 0 if undetermined
    float bpm_confidence  = 0.0f;  // 0..1
    int   key_pitch_class = -1;    // 0=C, 1=C#, ..., 11=B; -1 = undetermined
    bool  key_is_minor    = false; // only meaningful if key_pitch_class >= 0
    float key_strength    = 0.0f;  // 0..1
};

// mono: T samples, single channel, at `sample_rate` Hz. Returns false only
// for invalid input (null buffer, T<=0, sample_rate<=0); a low-confidence
// result on valid-but-hard-to-analyze audio is still `true` with low
// bpm_confidence/key_strength, not a failure.
bool analyze_mono(const float * mono, int T, int sample_rate, Result * out);

}  // namespace acebeat
