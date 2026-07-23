#pragma once
// safetensors-write.h: minimal safetensors writer (counterpart to safetensors.h)
//
// Format: 8 byte LE header length, JSON header, raw tensor data (row major,
// concatenated in entry order). Mirrors what PEFT/ComfyUI produce closely
// enough for src/adapter-merge.h to reload without any special casing.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct STWriteEntry {
    std::string           name;
    std::string            dtype;  // "F32", "BF16", "F16"
    std::vector<int64_t>  shape;   // row major, PyTorch convention
    const void *           data;
    size_t                 nbytes;
};

// Serializes entries plus an optional raw JSON object body (already
// formatted, no surrounding braces) to store under "__metadata__".
static bool st_write(const std::string & path, const std::vector<STWriteEntry> & entries,
                     const std::string & metadata_json_body = "") {
    std::string header = "{";

    if (!metadata_json_body.empty()) {
        header += "\"__metadata__\":{" + metadata_json_body + "},";
    }

    size_t offset = 0;
    for (size_t i = 0; i < entries.size(); i++) {
        const STWriteEntry & e = entries[i];
        header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
        for (size_t d = 0; d < e.shape.size(); d++) {
            header += std::to_string(e.shape[d]);
            if (d + 1 < e.shape.size()) {
                header += ",";
            }
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + e.nbytes) + "]}";
        offset += e.nbytes;
        if (i + 1 < entries.size()) {
            header += ",";
        }
    }
    header += "}";

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[Safetensors] Cannot create %s\n", path.c_str());
        return false;
    }

    uint64_t hdr_len = header.size();
    if (fwrite(&hdr_len, sizeof(hdr_len), 1, f) != 1) {
        fclose(f);
        return false;
    }
    if (fwrite(header.data(), 1, header.size(), f) != header.size()) {
        fclose(f);
        return false;
    }
    for (const auto & e : entries) {
        if (e.nbytes > 0 && fwrite(e.data, 1, e.nbytes, f) != e.nbytes) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}
