#include "key.h"

#include <algorithm>
#include <cmath>

namespace acebeat {

namespace {

struct Profile {
    float major[12];
    float minor[12];
};

// A small ensemble of published key-profile templates, each indexed by
// semitone distance above the tonic (0 = tonic, ..., 11 = major seventh).
// Rather than commit to a single "best" profile globally, every track is
// scored against all of them and whichever profile fits it best wins --
// this is literally the approach Faraldo, Jorda & Herrera propose in "A
// Multi-Profile Method for Key Estimation in EDM" (AES 142nd Convention,
// 2017): different tracks/sub-genres respond better to different
// profiles, so picking per-track beats picking one profile for the whole
// corpus. Measured on the Phase 3.5 real-corpus accuracy gate: a single
// EDMA profile alone gave no reliable improvement over Krumhansl-Kessler
// (~50-58% exact-key agreement either way); this ensemble is the
// structural change tried after per-profile tuning plateaued.
const Profile kProfiles[] = {
    // EDMA (Faraldo, Jorda & Herrera, 2017): corpus-derived, fit to
    // electronic dance music specifically.
    { { 1.00f, 0.2875f, 0.5020f, 0.4048f, 0.6050f, 0.5614f, 0.3205f, 0.7966f, 0.3159f, 0.4506f, 0.4202f, 0.3889f },
      { 1.00f, 0.3096f, 0.4415f, 0.5827f, 0.3262f, 0.4948f, 0.2889f, 0.7804f, 0.4328f, 0.2903f, 0.5331f, 0.3217f } },
    // Krumhansl & Kessler, "Cognitive Foundations of Musical Pitch", 1990:
    // classical psychoacoustic probe-tone profile.
    { { 6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f, 2.52f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f },
      { 6.33f, 2.68f, 3.52f, 5.38f, 2.60f, 3.53f, 2.54f, 4.75f, 3.98f, 2.69f, 3.34f, 3.17f } },
    // Shaath, "Estimation of Key in Digital Music Recordings", 2011: a
    // profile tuned for popular/electronic DJ material.
    { { 6.6f, 2.0f, 3.5f, 2.3f, 4.6f, 4.0f, 2.5f, 5.2f, 2.4f, 3.7f, 2.3f, 3.4f },
      { 6.5f, 2.7f, 3.5f, 5.4f, 2.6f, 3.5f, 2.5f, 5.2f, 4.0f, 2.7f, 4.3f, 3.2f } },
    // Temperley, "What's Key for Key? The Krumhansl-Schmuckler Key-Finding
    // Algorithm Reconsidered", Music Perception 17, 1999: revised profile
    // giving major/minor the same mean, removing K-K's inherent minor bias.
    { { 5.0f, 2.0f, 3.5f, 2.0f, 4.5f, 4.0f, 2.0f, 4.5f, 2.0f, 3.5f, 1.5f, 4.0f },
      { 5.0f, 2.0f, 3.5f, 4.5f, 2.0f, 4.0f, 2.0f, 4.5f, 3.5f, 2.0f, 1.5f, 4.0f } },
};
const int kNumProfiles = (int) (sizeof(kProfiles) / sizeof(kProfiles[0]));

float pearson_correlation(const float * a, const float * b, int n) {
    double mean_a = 0.0, mean_b = 0.0;
    for (int i = 0; i < n; i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= n;
    mean_b /= n;

    double num = 0.0, den_a = 0.0, den_b = 0.0;
    for (int i = 0; i < n; i++) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    double den = std::sqrt(den_a * den_b);
    if (den < 1e-12) {
        return 0.0f;
    }
    return (float) (num / den);
}

}  // namespace

KeyResult estimate_key(const std::vector<float> & chroma) {
    KeyResult result;
    if (chroma.size() != 12) {
        return result;
    }

    struct Candidate {
        int   tonic;
        bool  minor;
        float corr;
    };
    std::vector<Candidate> candidates;
    candidates.reserve((size_t) kNumProfiles * 24);

    for (int p = 0; p < kNumProfiles; p++) {
        for (int tonic = 0; tonic < 12; tonic++) {
            for (int mode = 0; mode < 2; mode++) {
                const float * profile = (mode == 0) ? kProfiles[p].major : kProfiles[p].minor;
                float         rotated[12];
                for (int pc = 0; pc < 12; pc++) {
                    // profile[i] is the expected weight for the pitch `i`
                    // semitones above the tonic; invert to read off the
                    // expected weight at absolute pitch class `pc`.
                    int i       = (pc - tonic + 12) % 12;
                    rotated[pc] = profile[i];
                }
                float corr = pearson_correlation(chroma.data(), rotated, 12);
                candidates.push_back({ tonic, mode == 1, corr });
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
               [](const Candidate & a, const Candidate & b) { return a.corr > b.corr; });

    result.pitch_class = candidates[0].tonic;
    result.is_minor     = candidates[0].minor;
    float top = candidates[0].corr;

    // (A discrete "dominant/tonic correction" -- overriding the top pick
    // with its subdominant-direction alternative (tonic-7) when the two
    // are a near-tie -- was tried here, re-tuned and validated against the
    // FULL ~1900-file corpus (not just a small sample) at tolerance=0.90.
    // The underlying diagnosis is solid (root detection lands a fifth
    // above the true tonic 4.3x more often than the reverse, 37% of all
    // non-exact misses), but the fix itself measured genuinely flat
    // (56.3% vs. 56.6% baseline exact-key, within the ~2.2% margin of
    // error at this sample size -- a real null result, not noise this
    // time). Likely explanation: a correlation-closeness heuristic alone
    // can't tell "this is a genuine dominant/tonic confusion" apart from
    // "this is a legitimate close tie on an already-correct answer" (many
    // real tracks spend real time on both I and V) -- it probably fixes
    // about as many correct answers as it breaks. Reverted. Resolving
    // this would need actual disambiguating evidence, not just a
    // correlation-margin rule -- see the bass-register signal explored
    // separately in acebeat.cpp.)
    // "Second place" must be a genuinely different (tonic, mode) answer,
    // not just a different profile scoring the *same* answer slightly
    // differently -- with several profiles pooled, the top-ranked
    // candidate is very often exactly that (all profiles agreeing), which
    // would make the margin look artificially small even when every
    // profile agrees on the key.
    float second = -1.0f;
    for (const Candidate & c : candidates) {
        if (c.tonic != result.pitch_class || c.minor != result.is_minor) {
            second = c.corr;
            break;
        }
    }
    if (second < 0.0f) {
        second = top;  // every profile agreed on the same answer -- maximal confidence
    }
    result.strength = std::clamp(top - second, 0.0f, 1.0f);

    return result;
}

}  // namespace acebeat
