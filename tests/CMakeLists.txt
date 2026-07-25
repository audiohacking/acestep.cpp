# test-philox: Philox RNG reference dumper.
add_executable(test-philox test-philox.cpp)
target_include_directories(test-philox PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(test-philox PRIVATE m)
endif()

# test-model-store: exercise ModelStore load/evict policies against real GGUFs.
add_executable(test-model-store test-model-store.cpp)
target_link_libraries(test-model-store PRIVATE acestep-core)
link_ggml_backends(test-model-store)

# test-lm-prompt: byte for byte check of Phase 2 LM prompt builders
# against the Qwen3 chat template reference.
add_executable(test-lm-prompt test-lm-prompt.cpp)
target_link_libraries(test-lm-prompt PRIVATE acestep-core)
link_ggml_backends(test-lm-prompt)

# test-lora-roundtrip: safetensors writer + adapter-merge LoRA round-trip
# (Phase 1 of native LoRA training, see ../TRAINING_DEV.md).
add_executable(test-lora-roundtrip test-lora-roundtrip.cpp)
target_link_libraries(test-lora-roundtrip PRIVATE acestep-core)
link_ggml_backends(test-lora-roundtrip)

# test-lora-train-smoke: LoRA training backward smoke test + overfit check
# (Phase 2 of native LoRA training, see ../TRAINING_DEV.md).
add_executable(test-lora-train-smoke test-lora-train-smoke.cpp)
target_link_libraries(test-lora-train-smoke PRIVATE acestep-core)
link_ggml_backends(test-lora-train-smoke)

# test-grad-compare: dumps a single training step's gradient (real sample,
# fixed LoRA init/noise/t) for cross-checking against Python. Diagnostic
# tool, see ../TRAINING_DEV.md.
add_executable(test-grad-compare test-grad-compare.cpp)
target_link_libraries(test-grad-compare PRIVATE acestep-core)
link_ggml_backends(test-grad-compare)

# gen-train-samples: synthetic sample tensor generator for testing
# `ace-train fit` end to end ahead of Phase 4 (`ace-train prepare`).
add_executable(gen-train-samples gen-train-samples.cpp)
target_include_directories(gen-train-samples PRIVATE ${CMAKE_SOURCE_DIR}/src)
link_ggml_backends(gen-train-samples)
