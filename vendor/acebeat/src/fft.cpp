#include "fft.h"

#include <cassert>
#include <cmath>

namespace acebeat {

int next_pow2(int n) {
    if (n <= 1) {
        return 1;
    }
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

void fft_forward_inplace(std::vector<std::complex<float>> & a) {
    const size_t n = a.size();
    if (n <= 1) {
        return;
    }
    assert((n & (n - 1)) == 0 && "fft_forward_inplace: size must be a power of 2");

    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    // Iterative Cooley-Tukey butterflies.
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * (float) M_PI / (float) len;
        const std::complex<float> wlen(cosf(ang), sinf(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; k++) {
                std::complex<float> u = a[i + k];
                std::complex<float> v = a[i + k + len / 2] * w;
                a[i + k]              = u + v;
                a[i + k + len / 2]    = u - v;
                w *= wlen;
            }
        }
    }
}

std::vector<float> hann_window(int n) {
    std::vector<float> w((size_t) n);
    if (n <= 1) {
        if (n == 1) {
            w[0] = 1.0f;
        }
        return w;
    }
    // Periodic (DFT-even) Hann: divide by n, not n-1. Matches the
    // convention most spectral-analysis code expects for windowed STFT
    // frames (as opposed to the "symmetric" Hann used for FIR filter design).
    for (int i = 0; i < n; i++) {
        w[(size_t) i] = 0.5f * (1.0f - cosf(2.0f * (float) M_PI * (float) i / (float) n));
    }
    return w;
}

std::vector<float> magnitude_spectrum(const std::vector<float> & frame, int fft_size) {
    int n = fft_size > 0 ? fft_size : next_pow2((int) frame.size());
    assert((n & (n - 1)) == 0 && "magnitude_spectrum: fft_size must be a power of 2");

    std::vector<std::complex<float>> buf((size_t) n, std::complex<float>(0.0f, 0.0f));
    size_t                           copy_n = std::min(frame.size(), (size_t) n);
    for (size_t i = 0; i < copy_n; i++) {
        buf[i] = std::complex<float>(frame[i], 0.0f);
    }

    fft_forward_inplace(buf);

    std::vector<float> mag((size_t) (n / 2 + 1));
    for (int k = 0; k <= n / 2; k++) {
        mag[(size_t) k] = std::abs(buf[(size_t) k]);
    }
    return mag;
}

}  // namespace acebeat
