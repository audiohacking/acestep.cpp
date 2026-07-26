#include "tempo.h"

#include <algorithm>
#include <cmath>

namespace acebeat {

TempoResult estimate_tempo(const std::vector<float> & envelope, float envelope_rate, float min_bpm, float max_bpm) {
    TempoResult result;
    if (envelope.size() < 4 || envelope_rate <= 0.0f || min_bpm <= 0.0f || max_bpm <= min_bpm) {
        return result;
    }

    // Mean-center for a cleaner autocorrelation (removes the DC bias that
    // would otherwise dominate every lag equally).
    double mean = 0.0;
    for (float v : envelope) {
        mean += v;
    }
    mean /= (double) envelope.size();
    std::vector<float> centered(envelope.size());
    for (size_t i = 0; i < envelope.size(); i++) {
        centered[i] = (float) ((double) envelope[i] - mean);
    }

    int min_lag = std::max(1, (int) std::lround(60.0f * envelope_rate / max_bpm));
    int max_lag = std::min((int) centered.size() - 2, (int) std::lround(60.0f * envelope_rate / min_bpm));
    if (max_lag <= min_lag) {
        return result;
    }

    // Autocorrelation over the valid lag range, normalized by overlap
    // length so longer lags (which have fewer overlapping samples) aren't
    // penalized relative to shorter ones -- otherwise a real periodicity
    // at a longer lag would systematically look weaker than a shorter one
    // for no musical reason.
    std::vector<float> ac((size_t) (max_lag - min_lag + 1));
    for (int lag = min_lag; lag <= max_lag; lag++) {
        double sum = 0.0;
        int    n   = (int) centered.size() - lag;
        for (int i = 0; i < n; i++) {
            sum += (double) centered[(size_t) i] * (double) centered[(size_t) (i + lag)];
        }
        ac[(size_t) (lag - min_lag)] = (float) (sum / (double) n);
    }

    // Autocorrelation of a periodic signal peaks near-equally at every
    // integer multiple of the true period (lag=T, 2T, 3T, ...): for a
    // near-ideal periodic signal (e.g. a click train) every multiple is
    // almost exactly as strong as the fundamental, while on real music the
    // slower harmonic (half tempo, a strong backbeat/half-time feel) is
    // very often the *larger* one. Either way, the fundamental almost
    // always still shows up as a genuine, only-somewhat-weaker local
    // maximum. Fix: scan local maxima from the shortest lag (fastest
    // tempo) upward and take the first one within `tolerance` of the
    // strongest candidate -- generous enough to catch a fundamental that's
    // been outweighed by its own half-tempo harmonic, but still requires a
    // real local maximum (not an arbitrary lag) to avoid picking noise.
    std::vector<int> local_max_idx;
    for (size_t i = 1; i + 1 < ac.size(); i++) {
        if (ac[i] > ac[i - 1] && ac[i] > ac[i + 1] && ac[i] > 0.0f) {
            local_max_idx.push_back((int) i);
        }
    }
    if (local_max_idx.empty()) {
        int best = 0;
        for (size_t i = 1; i < ac.size(); i++) {
            if (ac[i] > ac[(size_t) best]) {
                best = (int) i;
            }
        }
        local_max_idx.push_back(best);
    }

    float raw_best_val = 0.0f;
    for (int idx : local_max_idx) {
        raw_best_val = std::max(raw_best_val, ac[(size_t) idx]);
    }
    if (raw_best_val <= 0.0f) {
        return result;  // no positive periodicity found at all
    }

    const float tolerance  = 0.6f;
    int         chosen_idx = local_max_idx[0];
    for (int idx : local_max_idx) {
        if (ac[(size_t) idx] >= tolerance * raw_best_val) {
            chosen_idx = idx;  // shortest lag (local_max_idx is ascending) clearing the bar
            break;
        }
    }

    // Parabolic interpolation around the chosen peak for sub-sample lag precision.
    float peak_lag = (float) (chosen_idx + min_lag);
    if (chosen_idx > 0 && chosen_idx + 1 < (int) ac.size()) {
        float y0    = ac[(size_t) (chosen_idx - 1)];
        float y1    = ac[(size_t) chosen_idx];
        float y2    = ac[(size_t) (chosen_idx + 1)];
        float denom = y0 - 2.0f * y1 + y2;
        if (std::abs(denom) > 1e-9f) {
            peak_lag += 0.5f * (y0 - y2) / denom;
        }
    }
    if (peak_lag <= 0.0f) {
        return result;
    }

    result.bpm = 60.0f * envelope_rate / peak_lag;

    // Confidence: how far the chosen peak stands above the average
    // autocorrelation level across the whole search range. A clear,
    // isolated periodicity peak stands well above the mean; noise-like
    // envelopes (no real periodicity) barely do.
    double ac_mean = 0.0;
    for (float v : ac) {
        ac_mean += v;
    }
    ac_mean /= (double) ac.size();
    float chosen_val = ac[(size_t) chosen_idx];
    if (chosen_val > 0.0f) {
        result.confidence = std::clamp((float) (((double) chosen_val - ac_mean) / (double) chosen_val), 0.0f, 1.0f);
    }

    return result;
}

}  // namespace acebeat
