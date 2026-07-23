// gen-train-samples.cpp: generate synthetic sample tensor GGUFs for testing
// `ace-train fit` end to end before Phase 4 (`ace-train prepare`, real
// dataset decoration) exists. Not a pass/fail test: a data generator.
//
// Usage:
//   ./gen-train-samples --output <dir> --count N [--h-enc N] [--seed N]

#include "dit-train-data.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>

int main(int argc, char ** argv) {
    std::string output;
    int         count = 8;
    int         h_enc = 2048;  // matches the turbo/XL text encoder hidden size
    uint32_t    seed  = 42;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--count" && i + 1 < argc) {
            count = atoi(argv[++i]);
        } else if (arg == "--h-enc" && i + 1 < argc) {
            h_enc = atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = (uint32_t) atoi(argv[++i]);
        }
    }
    if (output.empty()) {
        fprintf(stderr, "Usage: %s --output <dir> [--count N] [--h-enc N] [--seed N]\n", argv[0]);
        return 1;
    }

    std::filesystem::create_directories(output);

    std::mt19937                     rng(seed);
    std::normal_distribution<float> latent_dist(0.0f, 1.0f);
    std::normal_distribution<float> enc_dist(0.0f, 0.1f);
    std::uniform_int_distribution<int> t_choice(0, 2);
    std::uniform_int_distribution<int> enc_choice(0, 1);

    for (int i = 0; i < count; i++) {
        int T     = 16 + t_choice(rng) * 16;  // 16, 32, 48
        int enc_S = 4 + enc_choice(rng) * 4;  // 4, 8

        DiTTrainSample s;
        s.T     = T;
        s.enc_S = enc_S;
        s.caption = "synthetic sample " + std::to_string(i);

        s.target_latents.resize((size_t) T * 64);
        for (float & v : s.target_latents) {
            v = latent_dist(rng);
        }

        // text2music context: silence (zeros) + mask=1.0
        s.context_latents.assign((size_t) T * 128, 0.0f);
        for (int ti = 0; ti < T; ti++) {
            for (int c = 0; c < 64; c++) {
                s.context_latents[(size_t) ti * 128 + 64 + c] = 1.0f;
            }
        }

        s.encoder_hidden.resize((size_t) h_enc * enc_S);
        for (float & v : s.encoder_hidden) {
            v = enc_dist(rng);
        }

        std::string path = output + "/sample_" + std::to_string(i) + ".gguf";
        if (!dit_train_sample_write(s, path)) {
            fprintf(stderr, "[gen-train-samples] FATAL: failed to write %s\n", path.c_str());
            return 1;
        }
        fprintf(stderr, "[gen-train-samples] wrote %s (T=%d, enc_S=%d)\n", path.c_str(), T, enc_S);
    }

    fprintf(stderr, "[gen-train-samples] wrote %d samples to %s\n", count, output.c_str());
    return 0;
}
