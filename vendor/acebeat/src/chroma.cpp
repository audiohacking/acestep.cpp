#include "chroma.h"

#include "fft.h"

#include <cmath>

namespace acebeat {

namespace {

struct Peak {
    float freq;
    float mag;
};

}  // namespace

std::vector<float> compute_chroma(const float * mono, int T, int sample_rate, const ChromaParams & params) {
    std::vector<float> chroma(12, 0.0f);
    if (!mono || T <= 0 || sample_rate <= 0 || params.frame_size <= 0 || params.hop_size <= 0) {
        return chroma;
    }

    int                 fft_size = next_pow2(params.frame_size);
    std::vector<float>  window   = hann_window(params.frame_size);
    std::vector<float>  frame((size_t) params.frame_size);
    std::vector<Peak>    peaks;

    // Pass 1: STFT + peak-picking only, collecting every peak's exact
    // (not-yet-quantized-to-a-pitch-class) frequency. Only accumulate
    // energy from local spectral peaks, not every bin in range (Fujishima,
    // "Realtime Chord Recognition of Musical Sound", 1999 -- the original
    // Pitch Class Profile paper -- picks peaks rather than summing raw
    // magnitude). Real polyphonic audio has broadband percussive/noise
    // energy spread across every bin; summing it all in would dilute the
    // actual tonal content the pitch-class signal depends on, whereas a
    // genuine musical pitch shows up as an actual local maximum.
    float bin_hz = (float) sample_rate / (float) fft_size;
    for (int start = 0; start + params.frame_size <= T; start += params.hop_size) {
        double energy = 0.0;
        for (int i = 0; i < params.frame_size; i++) {
            double s = (double) mono[start + i];
            energy += s * s;
        }
        energy /= (double) params.frame_size;
        if (energy < (double) params.silence_threshold) {
            continue;  // silence gate: don't let noise-floor frames dilute the profile
        }

        for (int i = 0; i < params.frame_size; i++) {
            frame[(size_t) i] = mono[start + i] * window[(size_t) i];
        }
        std::vector<float> mag = magnitude_spectrum(frame, fft_size);

        // (Spectral whitening -- dividing out a locally-smoothed magnitude
        // envelope before peak-picking, a published HPCP preprocessing
        // step (Gomez, 2006) meant to counter a track's overall timbral
        // tilt -- was tried here via a moving-average envelope at several
        // window widths (30/300/800 bins) and measured *worse* than no
        // whitening at every width (24-54% exact-key vs. this version's
        // ~57%). Reverted; noted here so it isn't retried identically.)
        float frame_max = 0.0f;
        for (float m : mag) {
            frame_max = std::max(frame_max, m);
        }
        if (frame_max <= 0.0f) {
            continue;
        }
        const float peak_threshold = 0.08f * frame_max;  // ignore noise-floor bumps

        for (size_t k = 1; k + 1 < mag.size(); k++) {  // k=0 is DC, no pitch meaning
            float freq = (float) k * bin_hz;
            if (freq < params.min_freq || freq > params.max_freq) {
                continue;
            }
            bool is_peak = mag[k] > mag[k - 1] && mag[k] >= mag[k + 1] && mag[k] >= peak_threshold;
            if (is_peak) {
                // Parabolic interpolation on log-magnitude (McAulay &
                // Quatieri's classic quadratic peak-interpolation formula)
                // to estimate the peak's true sub-bin frequency, instead of
                // just using this bin's nominal center frequency -- a real
                // peak almost never lands exactly on a bin center, and
                // using the raw bin frequency introduces up to half a
                // bin's worth of frequency error before the pitch-class
                // mapping even starts.
                float log_lo = log10f(mag[k - 1] + 1e-9f);
                float log_mid = log10f(mag[k] + 1e-9f);
                float log_hi = log10f(mag[k + 1] + 1e-9f);
                float denom   = log_lo - 2.0f * log_mid + log_hi;
                float delta   = (std::abs(denom) > 1e-9f) ? 0.5f * (log_lo - log_hi) / denom : 0.0f;
                float interp_freq = ((float) k + delta) * bin_hz;

                // sqrt-compress magnitude: without this, a few very loud
                // bass/kick peaks can dominate the whole profile and drown
                // out quieter but still musically-relevant harmonic
                // content -- amplitude compression before pitch-class
                // accumulation is a standard HPCP refinement for exactly
                // this reason.
                peaks.push_back({ interp_freq, sqrtf(mag[k]) });
            }
        }
    }

    if (peaks.empty()) {
        return chroma;
    }

    // Tuning-frequency estimation (Gomez, "Tonal Description of Music
    // Audio Signals", 2006): real recordings aren't always tuned to
    // exactly A440 (mastering pitch shifts, non-standard tuning
    // references), which would otherwise misalign every peak from its
    // true equal-tempered pitch class by a small constant offset and blur
    // the whole profile. Estimate that offset as the magnitude-weighted
    // circular mean of each peak's deviation from the nearest semitone
    // (circular because -0.49 and +0.49 semitones are actually close to
    // each other, both near a semitone boundary from opposite sides), then
    // correct every peak by it before assigning pitch classes below.
    double sum_cos = 0.0, sum_sin = 0.0;
    for (const Peak & p : peaks) {
        float  midi      = 12.0f * log2f(p.freq / 440.0f) + 69.0f;
        float  deviation = midi - roundf(midi);  // in [-0.5, 0.5)
        double angle     = 2.0 * M_PI * (double) deviation;
        sum_cos += (double) p.mag * std::cos(angle);
        sum_sin += (double) p.mag * std::sin(angle);
    }
    float tuning_offset = (float) (std::atan2(sum_sin, sum_cos) / (2.0 * M_PI));  // semitones

    // Pass 2: assign each peak (and its sub-harmonics) to a pitch class
    // using the corrected pitch mapping -- no new FFTs needed, reusing the
    // peak list collected above.
    //
    // (A higher-resolution HPCP variant -- several bins per semitone with
    // a cosine-squared weighting window, folded down to 12 bins -- was
    // tried here and measured *worse* on the Phase 3.5 real-corpus gate
    // (33-48% exact-key vs. this version's ~57%) at every window width
    // tried. Combined with 8x sub-harmonic summation, the extra spreading
    // over-smooths the profile and washes out the contrast that makes
    // template correlation decisive. Reverted; noted here so it isn't
    // retried identically later.)
    for (const Peak & p : peaks) {
        // Sub-harmonic summation: a spectral peak at frequency f could be
        // the fundamental of the note actually sounding, or it could be a
        // harmonic of a lower note whose own fundamental is weak or
        // filtered out. Crediting every candidate fundamental
        // (f, f/2, f/3, ..., f/8) with decreasing weight means a real
        // note's overtone series reinforces its own pitch class across
        // many observed peaks, instead of each harmonic just adding noise
        // at its own (different) pitch class -- this is what makes real
        // HPCP-style key detection much more decisive than naive per-bin
        // pitch-class summation.
        for (int h = 1; h <= 8; h++) {
            float candidate_freq = p.freq / (float) h;
            if (candidate_freq < 20.0f) {  // below any plausible musical fundamental
                break;
            }
            float midi = 12.0f * log2f(candidate_freq / 440.0f) + 69.0f - tuning_offset;

            // Soft (triangular) binning instead of hard rounding: a peak
            // rarely lands exactly on a semitone center, and a peak that's
            // genuinely near the boundary between two pitch classes
            // shouldn't have its whole weight coin-flipped onto whichever
            // side it happens to round to -- split it between the two
            // nearest classes proportional to closeness instead. This
            // matters more here than in a single-peak-per-note scenario
            // because sub-harmonic summation already stacks many
            // (h, deviation) pairs per note; hard-rounding noise on each
            // one compounds.
            float midi_floor = floorf(midi);
            float frac        = midi - midi_floor;
            int   pc_lo        = ((int) midi_floor) % 12;
            if (pc_lo < 0) {
                pc_lo += 12;
            }
            int pc_hi = (pc_lo + 1) % 12;

            float weight = p.mag / (float) h;
            chroma[(size_t) pc_lo] += weight * (1.0f - frac);
            chroma[(size_t) pc_hi] += weight * frac;
        }
    }

    float total = 0.0f;
    for (float v : chroma) {
        total += v;
    }
    if (total > 0.0f) {
        for (float & v : chroma) {
            v /= total;
        }
    }
    return chroma;
}

}  // namespace acebeat
