#pragma once
// key.h: musical key estimation from a chroma profile (see chroma.h) via
// correlation against an ensemble of published major/minor key-profile
// templates (EDMA, Krumhansl-Kessler, Shaath, Temperley -- see key.cpp for
// full citations), picking whichever profile fits each track best rather
// than committing to one profile for the whole corpus. This mirrors the
// approach in Faraldo, Jorda & Herrera, "A Multi-Profile Method for Key
// Estimation in EDM" (AES 142nd Convention, 2017): not derived from
// Essentia's or any other project's source or output.

#include <vector>

namespace acebeat {

struct KeyResult {
    int   pitch_class = -1;     // 0=C, 1=C#, ..., 11=B; -1 = undetermined
    bool  is_minor     = false;
    float strength      = 0.0f; // 0..1, normalized top-vs-runner-up correlation margin
};

// chroma: a 12-element pitch-class profile, e.g. from compute_chroma().
KeyResult estimate_key(const std::vector<float> & chroma);

}  // namespace acebeat
