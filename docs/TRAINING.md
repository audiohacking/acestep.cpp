# LoRA Training Tutorial

`ace-train` trains LoRA adapters for the ACE-Step 1.5 DiT natively in C++,
with no Python and no PyTorch involved. It's a **standalone tool that lives
alongside** `ace-lm`/`ace-synth`/`ace-server` — it loads the same GGUF models
from the same `--models <dir>`, and its output adapters load through the
exact same `--adapters <dir>` mechanism you already use for generation.
Training does not alter generation: if you never run `ace-train`, nothing
about `ace-lm`/`ace-synth`'s behavior changes.

This is a practical walkthrough. For the numerical/architectural background
(why things are implemented the way they are, what was tried and rejected,
precision notes) see [`TRAINING_DEV.md`](../TRAINING_DEV.md) — that file is a
development log, not a tutorial.

## Overview

Two steps, both using the `ace-train` binary:

```
ace-train prepare --models <dir> --dataset <audio_dir> --output <tensors_dir> [options]
ace-train fit     --models <dir> --tensors <tensors_dir> --output <adapter_dir> [options]
```

`prepare` reads your audio files, labels them (captions/bpm/key/etc, either
from sidecar files or auto-generated), and encodes each one through the
frozen VAE/text/lyric/timbre encoders once, caching the result as a GGUF per
sample. `fit` trains LoRA adapters (self-attn + cross-attn `q/k/v/o_proj`) on
top of those cached tensors — the heavy encoders never run again during
training, only the DiT forward/backward.

The result is a standard PEFT-format adapter (`adapter_model.safetensors` +
`adapter_config.json`), directly usable by `ace-synth --adapters` and
`ace-server`, with no conversion step.

## Step 1: Prepare your dataset

Put your audio files (`.wav` or `.mp3`) in a folder. Two ways to label them:

**A. Auto-label everything** (simplest, needs an LM model):

```
./ace-train prepare \
    --models ./models \
    --dataset ./my-songs \
    --output  ./my-songs-tensors \
    --lm-model acestep-5Hz-lm-4B-Q8_0.gguf
```

Each file with no caption gets one run through the understand pipeline
(caption, lyrics, bpm, keyscale, timesignature, language all filled in
automatically).

**B. Sidecar files** (full control), one `{name}.json` per audio file:

```json
{
  "caption": "An upbeat, instrumental nu-disco track with a funky bassline...",
  "lyrics": "[Instrumental]",
  "bpm": 122,
  "keyscale": "A minor",
  "timesignature": "4",
  "language": "unknown"
}
```

(`{name}.lyrics.txt`/`{name}.txt` and `{name}.caption.txt` also work for
just lyrics/caption; see `ace-train prepare`'s `--help` output.) Any field
still missing after the sidecar is read falls back to auto-label if
`--lm-model` is given.

**Trigger words** (recommended for style/single-song adapters): a short tag
prepended to every sample's caption, so the whole adapter is reliably
invoked by one word instead of depending on the caption's exact wording at
inference time:

```
./ace-train prepare ... --trigger-word mystyle01 --tag-position prepend
```

At inference, include `mystyle01` in the caption you send. Use
`--tag-position replace` for a single-song adapter where you don't want any
natural-language caption at all — the trigger word becomes the entire
prompt.

**Important**: at inference, invoke a trigger-worded adapter through
`ace-lm` with `"use_cot_caption": false` in the request, so the LM's
chain-of-thought caption rewriting doesn't mangle or drop an
unrecognized-word trigger before it reaches the DiT.

## Step 2: Train

```
./ace-train fit \
    --models  ./models \
    --tensors ./my-songs-tensors \
    --output  ./my-lora \
    --rank 64 --alpha 128 \
    --epochs 1500 --grad-accum 1 \
    --save-every 250 \
    --seed 42
```

Recommended starting point, from direct comparison against the Python
reference trainer:

| Flag | Recommendation | Why |
|---|---|---|
| `--rank` / `--alpha` | `64` / `128` | The CLI default (8/16) is an older, lower-capacity setting; 64/128 is what the current Python trainer defaults to and what produces an audible effect at `adapter_scale=1.0`. |
| `--epochs` | ~1000-1500 for a single-song/style dataset (1-3 samples) | Watch the logged EMA loss; a single-sample adapter around EMA 0.15-0.25 is usually the sweet spot — much lower starts overfitting the one sample. |
| `--grad-accum` | `1` for a single-sample dataset | With one sample, every epoch is already one optimizer step; higher grad-accum just averages a sample against itself. |
| `--save-every` | a fraction of `--epochs` | Saves multiple checkpoints so you can pick the best-sounding one rather than only the last. |

Everything else (`--lr`, `--weight-decay`, `--cfg-ratio`, `--max-grad-norm`,
`--warmup-steps`, `--seed`) defaults to values matched against the Python
trainer's own defaults; leave them unless you know you want to deviate.

Training logs one line per epoch:

```
[ace-train] epoch 1500/1500 step 1500/1500 lr=1.00e-06 loss=0.329051 ema=0.139417
```

`loss` is the single-step MSE for that epoch's own random noise/timestep
draw (expect it to bounce around, sometimes a lot — that's the flow-matching
objective's own variance, not instability); `ema` (decay 0.1) is the
smoothed trend to actually watch.

Checkpoints land at `<output>/checkpoints/epoch_N/` and a final one at
`<output>/final/`, each a complete PEFT adapter directory.

## Step 3: Use your LoRA

Same two-step generation flow as always, just add the adapter:

```
# 1) LM generates audio codes from your caption (include the trigger word!)
./ace-lm --models ./models --request request.json

# 2) DiT + VAE render the audio, with the adapter applied
./ace-synth --models ./models --request request0.json \
    --adapters ./my-lora/checkpoints
```

In `request.json`, set `"task_type": "text2music"` and do **not** set
`audio_codes` (setting it silently switches generation to `"cover"` mode,
which conditions heavily on reproducing a fixed reference's structure —
not what you want when judging an adapter's learned style from scratch).
In the post-`ace-lm` request (`request0.json`, which now has `audio_codes`
filled in), add:

```json
{
  "adapter": "epoch_1500",
  "adapter_scale": 1.0
}
```

`"adapter"` is the checkpoint subdirectory name under whatever `--adapters
<dir>` you passed to `ace-synth`. `ace-server`'s `--adapters <dir>` works
the same way, selectable from the WebUI.

## Design note: why this is a separate tool that touches no shared files

`ace-train` is intentionally its own binary, and its graph-building code
(`src/dit-train-graph.h`) is a **self-contained fork**, not a
parameterized extension, of the production DiT graph builder
(`src/dit-graph.h`, used by `ace-lm`/`ace-synth`/`ace-server` via
`src/dit-sampler.h`). `src/dit-graph.h` carries zero training-specific
code — no LoRA structs, no flags, no debug hooks — it is exactly the
upstream generation graph. Everything training needs (LoRA delta
application, the bf16-precision-matching op on the LoRA branch, gradient
graph sizing, diagnostic layer-by-layer tensor dumps) lives entirely in
`src/dit-train-graph.h`, included only by `ace-train` and the diagnostic
test tool.

This is a deliberate duplication tradeoff: the shared forward-pass
architecture (timestep embedding, self/cross-attention, MLP, per-layer
composition) is copied between the two files rather than factored into one
parameterized version, so that adding or changing training-only behavior
never requires touching — and never risks regressing — the code path every
other tool depends on. The cost is that a future change to the DiT
architecture in `src/dit-graph.h` needs to be manually ported into
`src/dit-train-graph.h` too; there is no compiler-enforced guarantee they
stay in sync, only the file-level comment in `dit-train-graph.h` documenting
the invariant.

## Troubleshooting

- **Adapter has no audible effect**: check `adapter_scale` isn't 0; check
  rank/alpha weren't left at the CLI's older 8/16 default; check you
  actually included the trigger word in the caption you're generating with.
- **Training loss looks worse/better than a reference number from another
  run**: per-epoch loss is computed against that step's own randomly-drawn
  noise and timestep, so two runs (even same seed, different RNG
  implementations) are two different random walks — compare trends (EMA)
  over many steps, not single numbers, and prefer listening to the actual
  output over chasing a loss value.
- **`ace-train prepare` skips a sample**: it had no caption and no
  `--lm-model` was given to auto-label it.
