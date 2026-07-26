// test-fft.cpp: correctness tests for the FFT/windowing primitives
// (../src/fft.h). Plain assert-based harness, no test framework dependency.

#include "../src/fft.h"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace acebeat;

// Independent O(N^2) direct DFT, written completely separately from
// fft.cpp (not reusing any of its code), used only to cross-check
// fft_forward_inplace on a small case.
static std::vector<std::complex<float>> direct_dft(const std::vector<std::complex<float>> & x) {
    size_t                            n = x.size();
    std::vector<std::complex<float>> X(n);
    for (size_t k = 0; k < n; k++) {
        std::complex<float> sum(0.0f, 0.0f);
        for (size_t j = 0; j < n; j++) {
            float ang = -2.0f * (float) M_PI * (float) k * (float) j / (float) n;
            sum += x[j] * std::complex<float>(cosf(ang), sinf(ang));
        }
        X[k] = sum;
    }
    return X;
}

static void test_direct_dft_cross_check() {
    std::vector<float> real6 = { 1.0f, 2.0f, -1.0f, 0.5f, 3.0f, -2.0f };
    int                 n     = next_pow2((int) real6.size());
    assert(n == 8);

    std::vector<std::complex<float>> a((size_t) n, std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < real6.size(); i++) {
        a[i] = std::complex<float>(real6[i], 0.0f);
    }
    std::vector<std::complex<float>> b = a;

    fft_forward_inplace(a);
    std::vector<std::complex<float>> ref = direct_dft(b);

    for (int k = 0; k < n; k++) {
        float err = std::abs(a[(size_t) k] - ref[(size_t) k]);
        assert(err < 1e-3f);
    }
    printf("[test-fft] direct DFT cross-check OK\n");
}

static void test_bin_aligned_sine() {
    // Full-length, integer-cycle sines: zero spectral leakage expected, so
    // dynamic range assertions can be strict.
    struct Case {
        int n;  // == fft size == input length
        int k;  // target bin
    };
    Case cases[] = { { 64, 5 }, { 128, 10 }, { 256, 30 }, { 512, 7 }, { 1024, 100 } };

    for (auto c : cases) {
        std::vector<float> x((size_t) c.n);
        for (int i = 0; i < c.n; i++) {
            x[(size_t) i] = sinf(2.0f * (float) M_PI * (float) c.k * (float) i / (float) c.n);
        }
        std::vector<float> mag = magnitude_spectrum(x, c.n);

        int peak = 0;
        for (size_t i = 1; i < mag.size(); i++) {
            if (mag[i] > mag[(size_t) peak]) {
                peak = (int) i;
            }
        }
        assert(std::abs(peak - c.k) <= 1);

        int far = (c.k + (int) mag.size() / 3) % (int) mag.size();
        if (far == 0) {
            far = 1;
        }
        float ratio_db = 20.0f * log10f(mag[(size_t) peak] / (mag[(size_t) far] + 1e-9f));
        assert(ratio_db > 40.0f);
    }
    printf("[test-fft] bin-aligned sine peak test (strict, full-length) OK\n");
}

static void test_nonpow2_zero_padding() {
    // 100-sample input, 9 cycles over the 100 samples (non-integer number
    // of cycles once zero-padded to 128) -- expected peak near, not
    // exactly at, round(9/100*128)=12, with real (but weaker than the
    // no-leakage case) dynamic range. Verifies the zero-padding path
    // itself works, not perfect bin isolation under leakage.
    const int input_len = 100;
    const int cycles    = 9;
    std::vector<float> x((size_t) input_len);
    for (int i = 0; i < input_len; i++) {
        x[(size_t) i] = sinf(2.0f * (float) M_PI * (float) cycles * (float) i / (float) input_len);
    }
    std::vector<float> mag = magnitude_spectrum(x);  // fft_size defaults to next_pow2(100)=128

    int expected_n = next_pow2(input_len);
    assert(expected_n == 128);
    assert(mag.size() == (size_t) (expected_n / 2 + 1));

    int peak = 0;
    for (size_t i = 1; i < mag.size(); i++) {
        if (mag[i] > mag[(size_t) peak]) {
            peak = (int) i;
        }
    }
    int expected_bin = (int) lroundf((float) cycles / (float) input_len * (float) expected_n);  // ~12
    assert(std::abs(peak - expected_bin) <= 2);

    int far = (expected_bin + (int) mag.size() / 3) % (int) mag.size();
    if (far == 0) {
        far = 1;
    }
    float ratio_db = 20.0f * log10f(mag[(size_t) peak] / (mag[(size_t) far] + 1e-9f));
    assert(ratio_db > 15.0f);  // leakage present, but still a real, findable peak
    printf("[test-fft] non-power-of-2 zero-padding test OK (peak bin %d, expected ~%d)\n", peak, expected_bin);
}

static void test_parseval() {
    int                               n = 128;
    std::vector<std::complex<float>> x((size_t) n);
    for (int i = 0; i < n; i++) {
        x[(size_t) i] = std::complex<float>(sinf(0.3f * (float) i) + 0.5f * cosf(0.7f * (float) i), 0.0f);
    }
    std::vector<std::complex<float>> X = x;
    fft_forward_inplace(X);

    double time_energy = 0.0, freq_energy = 0.0;
    for (int i = 0; i < n; i++) {
        time_energy += std::norm(x[(size_t) i]);
    }
    for (int i = 0; i < n; i++) {
        freq_energy += std::norm(X[(size_t) i]);
    }
    freq_energy /= n;

    double rel_err = std::abs(time_energy - freq_energy) / time_energy;
    assert(rel_err < 1e-3);
    printf("[test-fft] Parseval check OK\n");
}

static void test_impulse() {
    int                               n = 64;
    std::vector<std::complex<float>> x((size_t) n, std::complex<float>(0.0f, 0.0f));
    x[0] = std::complex<float>(1.0f, 0.0f);
    fft_forward_inplace(x);
    for (int i = 0; i < n; i++) {
        assert(std::abs(std::abs(x[(size_t) i]) - 1.0f) < 1e-3f);
    }
    printf("[test-fft] impulse -> flat spectrum OK\n");
}

static void test_hann_window() {
    for (int n : { 1, 2, 8, 257 }) {
        std::vector<float> w = hann_window(n);
        assert((int) w.size() == n);
        for (float v : w) {
            assert(v >= -1e-5f && v <= 1.0f + 1e-5f);
        }
        if (n == 1) {
            assert(std::abs(w[0] - 1.0f) < 1e-5f);
        } else {
            assert(w[0] < 1e-3f);  // periodic Hann: w[0] == 0 exactly (up to fp)
        }
    }
    printf("[test-fft] Hann window sanity checks OK\n");
}

int main() {
    test_direct_dft_cross_check();
    test_bin_aligned_sine();
    test_nonpow2_zero_padding();
    test_parseval();
    test_impulse();
    test_hann_window();
    printf("[test-fft] ALL TESTS PASSED\n");
    return 0;
}
