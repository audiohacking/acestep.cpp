cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# version.h: embed git commit hash into all binaries.
# runs on every build, only rewrites if the hash changed.
set(VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(version ALL
    COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUTPUT=${VERSION_OUTPUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${VERSION_OUTPUT}"
    COMMENT "Checking git version"
)

# pthread: required explicitly on older glibc (< 2.34) where libpthread
# is not merged into libc. Modern distros link it implicitly but aarch64
# and older x86_64 toolchains need the explicit dependency.
find_package(Threads REQUIRED)

# Suppress MSVC fopen/sprintf deprecation warnings
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Put executables and backend .so in the same directory (build root).
# Without this, ggml defaults to bin/ for .so but executables stay in root,
# and ggml_backend_load_all() can't find the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# DiT tensor names can exceed default GGML_MAX_NAME of 64
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite/etc with warn_unused_result on all platforms
# SYCL excluded: _FORTIFY_SOURCE swaps memcpy for __memcpy_chk, unresolved in device code
if(NOT MSVC AND NOT GGML_SYCL)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: cover Turing to Blackwell for distributed binaries.
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

# Local ggml patches (training-only additions -- see patches/README.md):
# applied to the submodule's working tree at configure time, never committed
# into the submodule's own git history. Idempotent (checked via `git apply
# --reverse --check`) so repeated cmake reconfigures don't fail on a patch
# that's already applied, and `git submodule update` / a fresh checkout
# always needs this step to re-run once before the ggml sources it patches
# are compiled.
file(GLOB ACESTEP_GGML_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-*.patch")
foreach(_patch ${ACESTEP_GGML_PATCHES})
    execute_process(
        COMMAND git apply --reverse --check "${_patch}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ggml"
        RESULT_VARIABLE _ggml_patch_already_applied
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _ggml_patch_already_applied EQUAL 0)
        message(STATUS "Applying local ggml patch: ${_patch}")
        execute_process(
            COMMAND git apply "${_patch}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ggml"
            RESULT_VARIABLE _ggml_patch_apply_result
        )
        if(NOT _ggml_patch_apply_result EQUAL 0)
            message(FATAL_ERROR "Failed to apply local ggml patch: ${_patch}")
        endif()
    else()
        message(STATUS "Local ggml patch already applied: ${_patch}")
    endif()
endforeach()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
# CUDA graphs default on: standalone ggml ships them off. Overridable with
# -DGGML_CUDA_GRAPHS=OFF or at runtime with GGML_CUDA_DISABLE_GRAPHS=1.
if(NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()
add_subdirectory(ggml)

# cpp-httplib (HTTP server library, used by ace-server)
add_subdirectory(vendor/cpp-httplib)

# Shared compile options and ggml linkage
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
    )
    target_include_directories(${target} SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4505)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                              -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    endif()
    target_link_libraries(${target} PRIVATE ggml Threads::Threads)
    if(TARGET ggml-base)
        target_link_libraries(${target} PRIVATE ggml-base)
    endif()
    foreach(backend cpu blas cuda metal vulkan sycl)
        if(TARGET ggml-${backend})
            get_target_property(CURRENT_BACKEND_TYPE ggml-${backend} TYPE)
            if (CURRENT_BACKEND_TYPE STREQUAL "MODULE_LIBRARY")
                # DL mode: backend is loaded at runtime via dlopen,
                # skip all link-time deps.
                continue()
            endif()
            target_link_libraries(${target} PRIVATE ggml-${backend})
        endif()
    endforeach()

    # SYCL links its runtime PRIVATE, consumers need -fsycl to resolve libsycl.so
    if(TARGET ggml-sycl AND GGML_SYCL)
        target_link_options(${target} PRIVATE -fsycl)
    endif()
    add_dependencies(${target} version)
endmacro()

# yyjson (MIT, fast JSON parser/writer)
add_library(yyjson STATIC vendor/yyjson/yyjson.c)
target_include_directories(yyjson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
if(MSVC)
    target_compile_options(yyjson PRIVATE /W0)
else()
    target_compile_options(yyjson PRIVATE -w)
endif()

# Essentia (AGPL-3.0, vendored submodule) -- DSP-based BPM/key detection
# (RhythmExtractor2013, KeyExtractor) to complement the LM's own (unreliable)
# numeric-metadata guesses in ace-understand / ace-train prepare's auto-label.
# Built via its own waf build system in "lightweight" mode: only FFTW + libyaml
# as optional deps (no FFmpeg/libsamplerate/TagLib/Gaia/TensorFlow -- we do our
# own audio decode+resample and don't need file-metadata or ML classifiers).
#
# IMPORTANT: link this ONLY into the specific executables that use
# src/audio-analysis.h (currently ace-understand and ace-train), never into
# acestep-core. acestep-core is linked into every binary (ace-synth, ace-lm,
# ace-server too), and this project is MIT-licensed -- statically linking an
# AGPL-3.0 dependency into that shared library would carry AGPL's copyleft
# obligations onto binaries that never use it and have no need to.
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(ESSENTIA_YAML REQUIRED yaml-0.1)
pkg_check_modules(ESSENTIA_EIGEN3 REQUIRED eigen3)

include(ProcessorCount)
ProcessorCount(ESSENTIA_NPROC)
if(ESSENTIA_NPROC EQUAL 0)
    set(ESSENTIA_NPROC 4)
endif()

set(ESSENTIA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/essentia)
set(ESSENTIA_LIB ${ESSENTIA_DIR}/build/src/libessentia.a)
find_program(PYTHON3_EXECUTABLE python3 REQUIRED)
add_custom_command(
    OUTPUT ${ESSENTIA_LIB}
    COMMAND ${PYTHON3_EXECUTABLE} waf configure --lightweight=fftw,yaml --build-static --mode=release
    COMMAND ${PYTHON3_EXECUTABLE} waf build -j${ESSENTIA_NPROC}
    WORKING_DIRECTORY ${ESSENTIA_DIR}
    COMMENT "Building vendored Essentia (lightweight: fftw+yaml only)"
    VERBATIM
)
add_custom_target(essentia_build DEPENDS ${ESSENTIA_LIB})

add_library(essentia STATIC IMPORTED GLOBAL)
set_target_properties(essentia PROPERTIES IMPORTED_LOCATION ${ESSENTIA_LIB})
add_dependencies(essentia essentia_build)
target_include_directories(essentia INTERFACE
    ${ESSENTIA_DIR}/src
    ${ESSENTIA_DIR}/src/essentia
    ${ESSENTIA_DIR}/src/essentia/scheduler
    ${ESSENTIA_DIR}/src/essentia/streaming
    ${ESSENTIA_DIR}/src/essentia/streaming/algorithms
    ${ESSENTIA_DIR}/src/essentia/utils
    ${ESSENTIA_EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(essentia INTERFACE ${FFTW3F_LIBRARIES} ${ESSENTIA_YAML_LIBRARIES} Threads::Threads)

# Core library (shared between binaries)
add_library(acestep-core STATIC
    src/request.cpp
    src/model-store.cpp
    src/pipeline-lm.cpp
    src/pipeline-synth.cpp
    src/pipeline-synth-ops.cpp
    src/pipeline-understand.cpp
)
target_link_libraries(acestep-core PUBLIC yyjson)
link_ggml_backends(acestep-core)

# ace-synth: full pipeline (text-enc + cond + dit + vae + wav)
add_executable(ace-synth tools/ace-synth.cpp)
target_link_libraries(ace-synth PRIVATE acestep-core)
link_ggml_backends(ace-synth)

# ace-lm: LLM inference (CoT + audio codes)
add_executable(ace-lm tools/ace-lm.cpp)
target_link_libraries(ace-lm PRIVATE acestep-core)
link_ggml_backends(ace-lm)

# ace-train: native LoRA training (fit) + dataset preparation (prepare, Phase 4)
# Links essentia directly (not via acestep-core) -- see the AGPL note above.
add_executable(ace-train tools/ace-train.cpp)
target_link_libraries(ace-train PRIVATE acestep-core essentia)
link_ggml_backends(ace-train)

# webui: convert tools/webui/public/index.html.gz to a C header for embedding.
# the .gz is committed to git so the C++ build works without npm.
# to update: cd tools/webui && npm install && npm run build, then rebuild ace-server.
set(WEBUI_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/tools/public/index.html.gz")
set(WEBUI_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/index.html.gz.hpp")
add_custom_command(
    OUTPUT "${WEBUI_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${WEBUI_INPUT}" "-DOUTPUT=${WEBUI_OUTPUT}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/xxd.cmake"
    DEPENDS "${WEBUI_INPUT}"
    COMMENT "Embedding webui into index.html.gz.hpp"
)
set_source_files_properties(${WEBUI_OUTPUT} PROPERTIES GENERATED TRUE)

# ace-server: HTTP server (LM + synth endpoints + embedded webui)
add_executable(ace-server tools/ace-server.cpp ${WEBUI_OUTPUT})
target_include_directories(ace-server PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ace-server PRIVATE acestep-core httplib)
link_ggml_backends(ace-server)

# Links essentia directly (not via acestep-core) -- see the AGPL note above.
add_executable(ace-understand tools/ace-understand.cpp)
target_link_libraries(ace-understand PRIVATE acestep-core essentia)
link_ggml_backends(ace-understand)

# quantize: GGUF requantizer (BF16 -> K-quants)
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

# neural-codec: Oobleck VAE neural audio codec (encode/decode WAV <-> latent)
add_executable(neural-codec tools/neural-codec.cpp)
link_ggml_backends(neural-codec)

# mp3-codec: MP3 encoder/decoder (standalone, no ggml needed)
# The mp3/ headers are header-only and usable by ace-synth too via #include "mp3/mp3enc.h"
add_executable(mp3-codec tools/mp3-codec.cpp)
target_include_directories(mp3-codec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(mp3-codec version)
if(MSVC)
    target_compile_options(mp3-codec PRIVATE /W4 /wd4100 /wd4505)
else()
    target_compile_options(mp3-codec PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    target_link_libraries(mp3-codec PRIVATE m)
endif()
target_link_libraries(mp3-codec PRIVATE Threads::Threads)

add_subdirectory(tests)
