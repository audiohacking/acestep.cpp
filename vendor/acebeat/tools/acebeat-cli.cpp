// acebeat-cli: standalone BPM/key detection CLI. Builds and runs with
// zero acestep.cpp headers or libraries -- proof that vendor/acebeat/ is
// genuinely self-contained and could be extracted into its own repo
// without rework. Decoding is handled by this CLI's own tiny
// audio-decode.h, never by the acebeat library itself (see ../README.md).
//
// Usage:
//   acebeat-cli <file> [file...] [--json]
//
// Multiple files (including shell-expanded globs, e.g. acebeat-cli
// music/*.mp3) run in batch mode. With --json, each file's result prints
// as one JSON object per line (JSONL), so batch output is easy to
// pipe/parse -- this is what makes Phase 3.5's real-corpus accuracy-gate
// comparison reproducible and scriptable as a permanent part of the tree.

#include "../include/acebeat/acebeat.h"
#include "audio-decode.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char * kPitchNames[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s <file> [file...] [--json]\n"
            "  Detects BPM and musical key. Multiple files (or a shell glob) run in\n"
            "  batch mode. WAV (PCM16/Float32) and MP3 input.\n"
            "  --json   print one JSON object per line instead of human-readable text\n",
            prog);
}

static void print_result(const char * path, const acebeat::Result & r, bool ok, bool json) {
    const char * key = (ok && r.key_pitch_class >= 0 && r.key_pitch_class < 12) ? kPitchNames[r.key_pitch_class]
                                                                                  : "?";
    if (json) {
        if (!ok) {
            printf("{\"file\": \"%s\", \"error\": \"decode or analysis failed\"}\n", path);
            return;
        }
        printf(
            "{\"file\": \"%s\", \"bpm\": %.2f, \"bpm_confidence\": %.3f, \"key\": \"%s\", \"scale\": \"%s\", "
            "\"key_strength\": %.3f}\n",
            path, r.bpm, r.bpm_confidence, key, r.key_is_minor ? "minor" : "major", r.key_strength);
        return;
    }

    printf("%s\n", path);
    if (!ok) {
        printf("  ERROR: decode or analysis failed\n\n");
        return;
    }
    printf("  bpm: %.2f (confidence: %.2f)\n", r.bpm, r.bpm_confidence);
    printf("  key: %s %s (strength: %.2f)\n\n", key, r.key_is_minor ? "minor" : "major", r.key_strength);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    bool                       json = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--json")) {
            json = true;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            files.emplace_back(argv[i]);
        }
    }

    if (files.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    int failures = 0;
    for (const std::string & path : files) {
        int                  sample_rate = 0;
        std::vector<float> mono         = acebeat_cli::read_audio_mono(path.c_str(), &sample_rate);

        acebeat::Result r;
        bool              ok = !mono.empty() &&
                       acebeat::analyze_mono(mono.data(), (int) mono.size(), sample_rate, &r);
        if (!ok) {
            failures++;
        }
        print_result(path.c_str(), r, ok, json);
    }

    return failures > 0 ? 1 : 0;
}
