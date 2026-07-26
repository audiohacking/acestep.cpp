// test-tempo.cpp: correctness tests for onset envelope + tempo estimation
// (../src/onset.h, ../src/tempo.h), using synthetic click trains with
// known ground-truth BPM (no real audio files needed).

#include "../src/onset.h"
#include "../src/tempo.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

using namespace acebeat;

static const int SAMPLE_RATE = 44100;

// A train of short broadband clicks at the given BPM, with per-beat timing
// jitter (a fixed jitter fraction of the beat interval, not itself
// periodic) so the signal isn't a mathematically perfect comb -- closer to
// real (if extremely simplified) rhythmic audio than an idealized impulse
// train, without needing a real music file.
static std::vector<float> generate_click_train(float bpm, float duration_sec, float jitter_frac, unsigned seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> jitter_dist(-jitter_frac, jitter_frac);

    int                 T = (int) (duration_sec * SAMPLE_RATE);
    std::vector<float> x((size_t) T, 0.0f);

    float interval = 60.0f / bpm * (float) SAMPLE_RATE;  // samples per beat
    float pos       = 0.0f;
    while (pos < (float) T) {
        int click_pos = (int) std::lround(pos);
        for (int i = 0; i < 12 && click_pos + i < T; i++) {
            // Short decaying alternating-sign burst: broadband spectral
            // content (needed for spectral flux to register it), decays
            // to ~0 within ~12 samples so consecutive clicks don't blur
            // together even at the fastest tested tempo.
            x[(size_t) (click_pos + i)] += expf(-0.5f * (float) i) * ((i % 2 == 0) ? 1.0f : -1.0f);
        }
        pos += interval * (1.0f + jitter_dist(rng));
    }
    return x;
}

static void test_click_trains() {
    struct Case {
        float bpm;
    };
    Case cases[] = { { 60.0f }, { 90.0f }, { 120.0f }, { 128.0f }, { 140.0f }, { 175.0f } };

    OnsetParams params;  // defaults: frame_size=2048, hop_size=512
    float       envelope_rate = (float) SAMPLE_RATE / (float) params.hop_size;

    for (auto c : cases) {
        std::vector<float> x = generate_click_train(c.bpm, /*duration_sec=*/20.0f, /*jitter_frac=*/0.03f,
                                                     /*seed=*/(unsigned) (c.bpm * 100));
        std::vector<float> env = onset_envelope(x.data(), (int) x.size(), params);
        assert(!env.empty());

        TempoResult r = estimate_tempo(env, envelope_rate);
        assert(r.bpm > 0.0f);

        float rel_err = std::abs(r.bpm - c.bpm) / c.bpm;
        printf("[test-tempo] true=%.1f BPM  detected=%.2f BPM  (rel_err=%.2f%%)  confidence=%.2f\n", c.bpm, r.bpm,
               rel_err * 100.0f, r.confidence);
        assert(rel_err < 0.02f);        // within +/-2%
        assert(r.confidence > 0.3f);    // a clean click train should be a clearly confident case
    }
    printf("[test-tempo] all click-train BPM cases within tolerance OK\n");
}

static void test_silence_low_confidence() {
    // No periodicity at all (silence): must not fabricate a confident answer.
    std::vector<float> x((size_t) (5 * SAMPLE_RATE), 0.0f);
    OnsetParams         params;
    std::vector<float>  env = onset_envelope(x.data(), (int) x.size(), params);
    float                envelope_rate = (float) SAMPLE_RATE / (float) params.hop_size;
    TempoResult          r             = estimate_tempo(env, envelope_rate);
    assert(r.confidence < 0.3f);
    printf("[test-tempo] silence -> low confidence (%.2f) OK\n", r.confidence);
}

int main() {
    test_click_trains();
    test_silence_low_confidence();
    printf("[test-tempo] ALL TESTS PASSED\n");
    return 0;
}
