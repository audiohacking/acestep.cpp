#pragma once
// audio-decode.h: minimal, self-contained WAV/MP3 -> mono float decoder
// for acebeat-cli only. Deliberately not shared with the acebeat library
// itself (see ../README.md: acebeat never decodes audio) and deliberately
// not reusing acestep.cpp's own audio-io.h -- this keeps the CLI (and by
// extension this whole vendor/acebeat/ directory) buildable and useful
// with zero acestep.cpp headers, so it stays extractable into its own
// repo later without rework.
//
// WAV: PCM16 or Float32, mono or stereo, any sample rate. Assumes the
// common chunk order (fmt before data) real-world WAV files use.
// MP3: decoded via a vendored copy of minimp3 (CC0, ../third_party).

#define MINIMP3_IMPLEMENTATION
#include "../third_party/minimp3.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace acebeat_cli {

inline std::vector<float> read_wav_mono(const char * path, int * sample_rate) {
    std::vector<float> mono;
    *sample_rate = 0;

    FILE * f = fopen(path, "rb");
    if (!f) {
        return mono;
    }

    char riff[4], wave[4];
    uint32_t riff_size;
    if (fread(riff, 1, 4, f) != 4 || fread(&riff_size, 4, 1, f) != 1|| fread(wave, 1, 4, f) != 4 ||
        memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        return mono;
    }

    uint16_t format = 0, channels = 0, bits_per_sample = 0;
    uint32_t sr = 0;
    std::vector<uint8_t> data;

    char     chunk_id[4];
    uint32_t chunk_size;
    while (fread(chunk_id, 1, 4, f) == 4 && fread(&chunk_size, 4, 1, f) == 1) {
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            uint8_t fmt_buf[16];
            if (chunk_size < 16 || fread(fmt_buf, 1, 16, f) != 16) {
                break;
            }
            memcpy(&format, fmt_buf + 0, 2);
            memcpy(&channels, fmt_buf + 2, 2);
            memcpy(&sr, fmt_buf + 4, 4);
            memcpy(&bits_per_sample, fmt_buf + 14, 2);
            if (chunk_size > 16) {
                fseek(f, (long) (chunk_size - 16), SEEK_CUR);
            }
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data.resize(chunk_size);
            if (fread(data.data(), 1, chunk_size, f) != chunk_size) {
                break;
            }
        } else {
            fseek(f, (long) chunk_size, SEEK_CUR);
        }
        if (chunk_size % 2 == 1) {
            fseek(f, 1, SEEK_CUR);  // chunks are word-aligned
        }
    }
    fclose(f);

    if (data.empty() || channels == 0 || sr == 0) {
        return mono;
    }

    int T = 0;
    if (format == 1 && bits_per_sample == 16) {  // PCM16
        const int16_t * samples = reinterpret_cast<const int16_t *>(data.data());
        T                        = (int) (data.size() / sizeof(int16_t) / channels);
        mono.resize((size_t) T);
        for (int t = 0; t < T; t++) {
            double sum = 0.0;
            for (int c = 0; c < channels; c++) {
                sum += (double) samples[(size_t) t * channels + c] / 32768.0;
            }
            mono[(size_t) t] = (float) (sum / channels);
        }
    } else if (format == 3 && bits_per_sample == 32) {  // IEEE float32
        const float * samples = reinterpret_cast<const float *>(data.data());
        T                       = (int) (data.size() / sizeof(float) / channels);
        mono.resize((size_t) T);
        for (int t = 0; t < T; t++) {
            double sum = 0.0;
            for (int c = 0; c < channels; c++) {
                sum += (double) samples[(size_t) t * channels + c];
            }
            mono[(size_t) t] = (float) (sum / channels);
        }
    } else {
        return mono;  // unsupported format (e.g. PCM8, PCM24, ADPCM)
    }

    *sample_rate = (int) sr;
    return mono;
}

inline std::vector<float> read_mp3_mono(const char * path, int * sample_rate) {
    std::vector<float> mono;
    *sample_rate = 0;

    FILE * f = fopen(path, "rb");
    if (!f) {
        return mono;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        return mono;
    }
    std::vector<uint8_t> data((size_t) size);
    if (fread(data.data(), 1, (size_t) size, f) != (size_t) size) {
        fclose(f);
        return mono;
    }
    fclose(f);

    mp3dec_t dec;
    mp3dec_init(&dec);

    int    out_sr  = 0;
    int    out_nch = 0;
    size_t offset  = 0;
    while (offset < data.size()) {
        mp3dec_frame_info_t info;
        short                pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
        int samples = mp3dec_decode_frame(&dec, data.data() + offset, (int) (data.size() - offset), pcm, &info);
        if (info.frame_bytes == 0) {
            break;
        }
        offset += (size_t) info.frame_bytes;
        if (samples > 0) {
            if (out_sr == 0) {
                out_sr  = info.hz;
                out_nch = info.channels;
            }
            size_t base = mono.size();
            mono.resize(base + (size_t) samples);
            for (int t = 0; t < samples; t++) {
                double sum = 0.0;
                for (int c = 0; c < out_nch; c++) {
                    sum += (double) pcm[(size_t) t * (size_t) out_nch + (size_t) c] / 32768.0;
                }
                mono[base + (size_t) t] = (float) (sum / out_nch);
            }
        }
    }

    if (mono.empty() || out_sr == 0) {
        mono.clear();
        return mono;
    }
    *sample_rate = out_sr;
    return mono;
}

// Decodes a WAV or MP3 file (by extension) directly to mono float samples
// at the file's native sample rate. Returns an empty vector on failure.
inline std::vector<float> read_audio_mono(const char * path, int * sample_rate) {
    size_t len = strlen(path);
    bool   is_wav = len >= 4 && strcasecmp(path + len - 4, ".wav") == 0;
    return is_wav ? read_wav_mono(path, sample_rate) : read_mp3_mono(path, sample_rate);
}

}  // namespace acebeat_cli
