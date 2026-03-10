#!/bin/bash
# Lego mode: generate a guitar track over a backing track.
# Requires: acestep-v15-base model (turbo/sft do not support lego).
# Replace backing-track.wav with your source audio (WAV or MP3).

set -eu

../build/dit-vae \
    --src-audio backing-track.wav \
    --lego guitar \
    --request lego.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-base-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
