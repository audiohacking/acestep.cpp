// diag-quant-erasure.cpp: one-off diagnostic (not wired into CMakeLists).
// Question: does merging a REAL trained LoRA checkpoint into a Q8_0 DiT
// actually change the quantized bytes, or does the dequantize->add->
// requantize round-trip erase a delta this small? Loads the real DiT GGUF
// and a real trained adapter, runs the production adapter_merge() path,
// and reports how much the dequantized merged weight differs from the
// dequantized original for several real projections.
//
// Build ad hoc:
//   g++ -std=c++17 -I../src -I../ggml/include diag-quant-erasure.cpp \
//       -L../build -lggml-base -lggml-cpu -o /tmp/diag-quant-erasure
// (or just run via the project's normal cmake add_executable pattern)

#include "adapter-merge.h"
#include "ggml-cpu.h"
#include "gguf-weights.h"
#include "model-registry.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<float> dequant(const void * data, int64_t nel, ggml_type type) {
    std::vector<float> out((size_t) nel);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), data, (size_t) nel * sizeof(float));
        return out;
    }
    ggml_get_type_traits(type)->to_float(data, out.data(), nel);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <models_dir> <adapter_dir>\n", argv[0]);
        return 1;
    }
    std::string models_dir  = argv[1];
    std::string adapter_dir = argv[2];

    ModelRegistry reg;
    registry_scan(&reg, models_dir.c_str());
    std::string dit_path = reg.dit[0].path;
    fprintf(stderr, "DiT: %s\n", dit_path.c_str());

    GGUFModel gf;
    gf_load(&gf, dit_path.c_str());

    const char * names[] = {
        "decoder.layers.0.self_attn.q_proj.weight",  "decoder.layers.0.self_attn.o_proj.weight",
        "decoder.layers.0.cross_attn.q_proj.weight", "decoder.layers.12.self_attn.q_proj.weight",
        "decoder.layers.23.self_attn.q_proj.weight",
    };

    WeightCtx wctx;
    wctx_init(&wctx, 8);
    std::vector<const void *> base_ptrs;
    for (const char * n : names) {
        const void *          ptr   = gf_get_data(gf, n);
        struct ggml_tensor *  tmeta = ggml_get_tensor(gf.meta, n);
        size_t                nb    = ggml_row_size(tmeta->type, tmeta->ne[0]) * (size_t) tmeta->ne[1];
        base_ptrs.push_back(ptr);
        wctx.pending.push_back({ nullptr, ptr, nb, 0 });
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    bool ok = adapter_merge(&wctx, gf, adapter_dir.c_str(), 1.0f, backend);
    fprintf(stderr, "adapter_merge ok=%d\n", ok);

    for (size_t i = 0; i < 5; i++) {
        const char * n     = names[i];
        struct ggml_tensor * tmeta = ggml_get_tensor(gf.meta, n);
        ggml_type             type  = tmeta->type;
        int64_t               nel   = tmeta->ne[0] * tmeta->ne[1];

        std::vector<float> before = dequant(base_ptrs[i], nel, type);
        std::vector<float> after  = dequant(wctx.pending[i].src, nel, type);

        if (i == 0) {
            fprintf(stderr, "  before[0:8] =");
            for (int k = 0; k < 8; k++) fprintf(stderr, " %.8g", before[k]);
            fprintf(stderr, "\n  after[0:8]  =");
            for (int k = 0; k < 8; k++) fprintf(stderr, " %.8g", after[k]);
            fprintf(stderr, "\n  delta[0:8]  =");
            for (int k = 0; k < 8; k++) fprintf(stderr, " %.8g", after[k] - before[k]);
            fprintf(stderr, "\n");
        }

        double sum_abs_diff = 0, max_abs_diff = 0, sum_abs_before = 0;
        int    n_identical_bytes = 0;
        const uint8_t * b_before = (const uint8_t *) base_ptrs[i];
        const uint8_t * b_after  = (const uint8_t *) wctx.pending[i].src;
        size_t          nbytes   = wctx.pending[i].nbytes;
        for (size_t k = 0; k < nbytes; k++) {
            if (b_before[k] == b_after[k]) {
                n_identical_bytes++;
            }
        }
        for (int64_t k = 0; k < nel; k++) {
            double d = fabs((double) after[k] - (double) before[k]);
            sum_abs_diff += d;
            if (d > max_abs_diff) {
                max_abs_diff = d;
            }
            sum_abs_before += fabs((double) before[k]);
        }
        fprintf(stderr,
                "%s (type=%d, nel=%lld, nbytes=%zu):\n"
                "  identical bytes: %d / %zu (%.1f%%)\n"
                "  mean|delta|=%.6g  max|delta|=%.6g  mean|base|=%.6g  ratio(mean delta/mean base)=%.6g\n",
                n, (int) type, (long long) nel, nbytes, n_identical_bytes, nbytes,
                100.0 * n_identical_bytes / nbytes, sum_abs_diff / nel, max_abs_diff, sum_abs_before / nel,
                (sum_abs_diff / nel) / (sum_abs_before / nel + 1e-12));
    }

    return 0;
}
