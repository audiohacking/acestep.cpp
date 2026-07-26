// test-key.cpp: correctness tests for chroma extraction + key estimation
// (../src/chroma.h, ../src/key.h), using synthetic scale/chord constructions
// with known ground-truth key (no real audio files needed).

#include "../src/chroma.h"
#include "../src/key.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace acebeat;

static const int SAMPLE_RATE = 44100;
static const float TWO_PI    = 6.283185307179586f;

// midi 60 = C4; frequency of pitch class `pc` in the octave starting at C4.
static float pc_frequency(int pc) {
    int midi = 60 + pc;
    return 440.0f * powf(2.0f, (float) (midi - 69) / 12.0f);
}

// Synthesizes a signal from a set of (pitch_class, weight) tones -- a crude
// stand-in for "this key's diatonic scale, with the tonic triad emphasized
// the way real music emphasizes it," which is exactly the cue
// tonic-weighted chroma/key-template matching relies on.
static std::vector<float> synth_tones(const std::vector<std::pair<int, float>> & tones, float duration_sec) {
    int                 T = (int) (duration_sec * SAMPLE_RATE);
    std::vector<float> x((size_t) T, 0.0f);
    for (auto & [pc, weight] : tones) {
        float freq = pc_frequency(pc);
        for (int n = 0; n < T; n++) {
            x[(size_t) n] += weight * sinf(TWO_PI * freq * (float) n / (float) SAMPLE_RATE);
        }
    }
    return x;
}

struct KeyCase {
    const char *                       name;
    std::vector<std::pair<int, float>> tones;  // {pitch_class, weight}
    int                                expected_pc;
    bool                                expected_minor;
    float                               min_strength;  // relative-minor stress cases are supposed to be close calls
};

static void run_case(const KeyCase & c) {
    std::vector<float> x      = synth_tones(c.tones, /*duration_sec=*/2.0f);
    std::vector<float> chroma = compute_chroma(x.data(), (int) x.size(), SAMPLE_RATE);
    KeyResult           r      = estimate_key(chroma);

    printf("[test-key] %-30s expected pc=%2d minor=%d  got pc=%2d minor=%d strength=%.3f\n", c.name, c.expected_pc,
           c.expected_minor, r.pitch_class, r.is_minor, r.strength);

    assert(r.pitch_class == c.expected_pc);
    assert(r.is_minor == c.expected_minor);
    assert(r.strength > c.min_strength);  // must be a real win, not a coin-flip tie
}

int main() {
    // C major: scale C D E F G A B (0,2,4,5,7,9,11), tonic triad C-E-G emphasized.
    run_case({ "C major",
                { { 0, 3.0f }, { 2, 1.0f }, { 4, 2.0f }, { 5, 1.0f }, { 7, 2.0f }, { 9, 1.0f }, { 11, 1.0f } },
                0, false, 0.05f });

    // G major: scale G A B C D E F# (7,9,11,0,2,4,6), tonic triad G-B-D emphasized.
    run_case({ "G major",
                { { 7, 3.0f }, { 9, 1.0f }, { 11, 2.0f }, { 0, 1.0f }, { 2, 2.0f }, { 4, 1.0f }, { 6, 1.0f } },
                7, false, 0.05f });

    // A minor: SAME 7 pitch classes as C major (relative minor), but tonic
    // triad A-C-E emphasized instead -- the classic relative-major/minor
    // confusion stress case. Deliberately close: with several profiles
    // pooled (key.cpp), a real close call surfaces a smaller consensus
    // margin than a single profile alone would report, so the bar here is
    // just "not a literal tie," not "a clear win" like the major cases.
    run_case({ "A minor (relative of C major)",
                { { 9, 3.0f }, { 11, 1.0f }, { 0, 2.0f }, { 2, 1.0f }, { 4, 2.0f }, { 5, 1.0f }, { 7, 1.0f } },
                9, true, 0.01f });

    // E minor: SAME 7 pitch classes as G major (relative minor), tonic
    // triad E-G-B emphasized.
    run_case({ "E minor (relative of G major)",
                { { 4, 3.0f }, { 6, 1.0f }, { 7, 2.0f }, { 9, 1.0f }, { 11, 2.0f }, { 0, 1.0f }, { 2, 1.0f } },
                4, true, 0.01f });

    printf("[test-key] ALL TESTS PASSED\n");
    return 0;
}
