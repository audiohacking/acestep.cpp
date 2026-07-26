# acebeat: MIT-licensed BPM/key detection (see README.md). Plain library,
# no third-party dependencies -- deliberately, so its own licensing is
# trivial to audit at a glance.

cmake_minimum_required(VERSION 3.14)

# Detached-checkout guard, mirroring ggml/CMakeLists.txt's GGML_STANDALONE:
# lets this directory be built on its own (e.g. if it's ever pulled out
# into its own repo) without requiring anything from the parent project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(acebeat LANGUAGES CXX)
    set(ACEBEAT_STANDALONE ON)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
    set(ACEBEAT_STANDALONE OFF)
endif()

option(ACEBEAT_BUILD_TESTS "acebeat: build tests" ${ACEBEAT_STANDALONE})
option(ACEBEAT_BUILD_CLI   "acebeat: build the acebeat-cli tool" ${ACEBEAT_STANDALONE})

add_library(acebeat STATIC
    src/acebeat.cpp
    src/fft.cpp
    src/onset.cpp
    src/tempo.cpp
    src/chroma.cpp
    src/key.cpp
)
target_include_directories(acebeat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(acebeat PRIVATE /W4)
else()
    target_compile_options(acebeat PRIVATE -Wall -Wextra)
endif()

if(ACEBEAT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ACEBEAT_BUILD_CLI)
    add_subdirectory(tools)
endif()
