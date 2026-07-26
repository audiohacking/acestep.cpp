#pragma once
// onset.h: spectral-flux onset-strength envelope, the standard front-end
// for autocorrelation/comb-filter tempo estimation (see tempo.h). Written
// from the general published description of spectral flux onset detection
// (see e.g. Bello et al., "A Tutorial on Onset Detection in Music
// Signals," 2005) -- a decades-old, widely independently-reimplemented
// technique, not derived from any specific third-party implementation.

#include <vector>

namespace acebeat {

struct OnsetParams {
    int frame_size = 2048;  // samples per analysis frame
    int hop_size   = 512;   // samples between frame starts
};

// Returns one onset-strength value per hop (length = floor((T-frame_size)/hop_size),
// i.e. one fewer than the number of frames analyzed, since the first frame
// has no predecessor to diff against). Empty on invalid input.
std::vector<float> onset_envelope(const float * mono, int T, const OnsetParams & params = {});

}  // namespace acebeat
