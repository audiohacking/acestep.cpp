#pragma once
// fft.h: internal DSP primitives (FFT, windowing). Not part of the public
// API (see include/acebeat/acebeat.h) -- shared by onset/tempo and chroma/key.
//
// Iterative radix-2 Cooley-Tukey (Cooley & Tukey, 1965), forward transform
// only: magnitude spectra are all this library ever needs, never an
// inverse transform. Written from the standard textbook algorithm
// description, not derived from or compared against any third-party FFT
// implementation's source.

#include <complex>
#include <vector>

namespace acebeat {

// Smallest power of 2 >= n (n >= 1).
int next_pow2(int n);

// In-place forward FFT. data.size() MUST be a power of 2 (caller's
// responsibility; asserts in debug builds). Unnormalized (no 1/N scaling),
// matching the usual DSP-library convention.
void fft_forward_inplace(std::vector<std::complex<float>> & data);

// Periodic Hann window of length n (n >= 1), values in [0,1].
std::vector<float> hann_window(int n);

// Zero-pads `frame` to `fft_size` (or next_pow2(frame.size()) if
// fft_size<=0), runs fft_forward_inplace, and returns |X[k]| for
// k=0..fft_size/2 inclusive (the non-redundant half of a real-input FFT).
std::vector<float> magnitude_spectrum(const std::vector<float> & frame, int fft_size = 0);

}  // namespace acebeat
