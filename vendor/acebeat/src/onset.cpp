#include "onset.h"

#include "fft.h"

namespace acebeat {

std::vector<float> onset_envelope(const float * mono, int T, const OnsetParams & params) {
    std::vector<float> result;
    if (!mono || T <= 0 || params.frame_size <= 0 || params.hop_size <= 0) {
        return result;
    }

    int                 fft_size = next_pow2(params.frame_size);
    std::vector<float>  window   = hann_window(params.frame_size);
    std::vector<float>  frame((size_t) params.frame_size);
    std::vector<float>  prev_mag;

    for (int start = 0; start + params.frame_size <= T; start += params.hop_size) {
        for (int i = 0; i < params.frame_size; i++) {
            frame[(size_t) i] = mono[start + i] * window[(size_t) i];
        }
        std::vector<float> mag = magnitude_spectrum(frame, fft_size);

        if (!prev_mag.empty()) {
            float flux = 0.0f;
            for (size_t k = 0; k < mag.size(); k++) {
                float diff = mag[k] - prev_mag[k];
                if (diff > 0.0f) {
                    flux += diff;
                }
            }
            result.push_back(flux);
        }
        prev_mag = std::move(mag);
    }

    return result;
}

}  // namespace acebeat
