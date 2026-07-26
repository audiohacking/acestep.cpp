# LoRA Training in acestep.cpp — Development Tracker

Mission: native LoRA training for the ACE-Step 1.5 DiT, 0% Python.
Dataset decoration via the existing understand pipeline, tensor caching via
GGUF, training via GGML autograd + ggml-opt (AdamW), output as PEFT
safetensors directly loadable by the existing adapter registry.

Reference: [ACE-Step-1.5 LoRA Training Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md)
and the Python trainer (`acestep/training/trainer.py`, studied from a local
clone at `../acestep-megalora/acestep-repo`).

## The training recipe (from the Python reference)

Only the **DiT decoder** is trained, with LoRA on `q_proj, k_proj, v_proj,
o_proj` (self-attn + cross-attn). VAE, text encoder, cond encoder, LM are
frozen and only used for dataset preprocessing.

Per training step (turbo model, no CFG):

```
x0   = target_latents            # VAE-encoded audio [B, T, 64]
x1   = randn_like(x0)            # noise
t    ~ uniform choice from TURBO_SHIFT3_TIMESTEPS (8 discrete values)
       [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]
xt   = t * x1 + (1 - t) * x0     # flow-matching interpolation
v    = DiT(xt, t, t_r=t, encoder_hidden_states, context_latents, masks)
loss = MSE(v, x1 - x0)           # predict the flow field
```

Preprocessed tensors per sample (built once, cached):

| Tensor | Shape | Producer |
|---|---|---|
| `target_latents` | [T, 64] | VAE encoder on 48kHz stereo audio |
| `attention_mask` | [T] | all-ones up to true length |
| `encoder_hidden_states` | [L, D] | CondEncoder(text_hs, lyric_hs, timbre=zeros[1,64]) |
| `encoder_attention_mask` | [L] | from CondEncoder |
| `context_latents` | [T, 128] | concat(silence_latent[:T], ones-mask) — text2music context |

Text prompt for the text encoder is exactly the inference SFT prompt:
`# Instruction\n{DIT_INSTRUCTION}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>`
(already built at `src/pipeline-synth-ops.cpp:423`). Lyrics go through
`embed_tokens` only (max_length 512, padded); text prompt through the full
Qwen3 encoder (max_length 256, padded).

Hyperparameters (Python defaults):

| Param | Default |
|---|---|
| rank / alpha / dropout | 8 / 16 / 0.1 |
| target_modules | q_proj, k_proj, v_proj, o_proj |
| learning_rate | 1e-4 |
| optimizer | AdamW, weight_decay 0.01 |
| schedule | LinearLR warmup (0.1→1.0, ≤100 steps, capped at total/10) then cosine annealing to lr*0.01 |
| batch_size / grad_accum | 1 / 4 |
| max_grad_norm | 1.0 |
| epochs | ~100 default; tutorial suggests 500–800 for small sets |
| seed | 42 |
| precision | bf16 compute, fp32 LoRA params + optimizer state |

Checkpoint output: PEFT directory (`adapter_model.safetensors` +
`adapter_config.json`) — the exact format `src/adapter-merge.h` already
consumes. Loss is *not* masked over padded frames in Python (batch=1 makes
padding a non-issue there); we match by training at true length per sample.

## Feasibility audit (done 2026-07-23)

The GGML fork already contains everything needed to train:

- `ggml/include/ggml-opt.h`: full training module (AdamW + SGD, MSE loss,
  gradient accumulation). **Not used directly** — see Phase 2 below for why
  its high-level dataset/epoch API is unsafe for our variable-T graphs, and
  what we use instead (`ggml_build_backward_expand` + `ggml_opt_step_adamw`
  called directly, momentum tracked by our own parameter identity).
- `ggml_build_backward_expand` covers every op in the DiT training graph:
  ADD, SUB, MUL, MUL_MAT, SCALE, CONT, RESHAPE, VIEW, PERMUTE, TRANSPOSE,
  CPY/CAST, RMS_NORM, ROPE, SOFT_MAX (mask ok if mask needs no grads —
  ours doesn't), SILU, SWIGLU (split variant only), plus
  CROSS_ENTROPY/MSE losses.
- CUDA kernels exist for the backward-specific ops: `OUT_PROD`,
  `RMS_NORM_BACK`, `SOFT_MAX_BACK`, `SILU_BACK`, `OPT_STEP_ADAMW`.
- The DiT graph (`src/dit-graph.h`) already has the two fallback paths
  training requires:
  - no-flash-attn path (`soft_max_ext`) — FLASH_ATTN_EXT has **no backward**,
    so training always runs with `use_flash_attn = false`.
  - split-swiglu path (`ggml_swiglu_split`) — fused `ggml_swiglu` has no
    backward; training uses the non-fused FFN path.
- `TIMESTEP_EMBEDDING` has no backward but sits on the frozen temb/adaLN
  branch; no trainable param is upstream of it, so autograd never visits it.
- Gradients flow *through* frozen projections via
  `out_prod(W, transpose(grad))` — weights themselves get no grad
  (only LoRA A/B are `ggml_set_param`).

### Known constraints / blockers

1. **CUDA `OUT_PROD` is F32-only** (`ggml-cuda.cu:4819`). Backward through a
   frozen projection references its weight tensor directly, so every weight
   on the backward path must be F32 in the training graph. Options:
   a. Dequantize DiT weights to F32 at load for training (~8.6 GB for the
      2B DiT; `adapter-merge.h` already has dequant machinery). Simplest.
   b. Insert `ggml_cast(W→F32)` nodes in the training graph; keeps weights
      BF16-resident, casts transiently per layer (allocator reuses buffers).
   c. Fork patch: BF16 support in the out_prod kernels.
   Start with (a), measure, then optimize. Python also refuses to train on
   quantized weights, so "train needs an unquantized DiT GGUF" is accepted UX.
2. **No gradient checkpointing in ggml-opt.** Activation memory for
   24 layers × full backward at T≈2000+ latent frames may exceed consumer
   VRAM. Mitigations: cap/segment training sample duration (chunk long songs,
   like the tutorial recommends anyway), batch=1 + grad accumulation,
   CPU-backend fallback. Measure in Phase 2; checkpointing (recompute per
   layer via two-pass scheduling) is a stretch goal.
3. **LoRA dropout**: no dropout op in GGML. Skipped initially (inference-time
   equivalent; acceptable for rank-8 adapters). Optional later via
   Bernoulli mask input generated host-side per step (philox.h exists).
4. **Fused QKV**: model-store fuses Q/K/V at load. Training graph must keep
   LoRA branches per projection: base fused matmul + per-projection
   `B(A(x)) * (alpha/r)` added into the q/k/v slices (views) — or simply run
   the unfused path during training. Decide when writing the graph.
5. **safetensors write**: `src/safetensors.h` is read-only today. Need a
   small writer (JSON header + raw F32 payload) for `adapter_model.safetensors`
   + `adapter_config.json` emit.

## Design

`ace-train` (`tools/ace-train.cpp`), two subcommands sharing the models
dir. Both implemented (Phase 3: `fit`, Phase 4: `prepare`):

```
# 1) Dataset preparation: decorate + encode
./ace-train prepare \
    --models models \
    --dataset  path/to/audio_dir \        # .wav/.mp3 (+optional sidecars)
    --output   path/to/tensors_dir \      # per-sample GGUF
    --lm-model acestep-5Hz-lm-4B-Q8_0.gguf \ # optional, auto-labels missing captions
    --trigger-word mystyle01 --tag-position prepend  # optional, see Phase 5 below

# At inference, invoke it through ace-lm with use_cot_caption:false so the
# LM's Phase 1 CoT doesn't rewrite the (possibly unrecognized) trigger word
# before it reaches the DiT -- see Phase 5's Decisions log entry.

# 2) Training
./ace-train fit \
    --models   models \
    --tensors  path/to/tensors_dir \
    --output   adapters/my-lora \
    --rank 8 --alpha 16 --lr 1e-4 --epochs 100 --grad-accum 4 \
    --save-every 10 --seed 42 [--val-split 0.1]
```

### prepare stage (implemented, Phase 4)

Per audio file in `--dataset` (`src/dit-prepare.h`):
1. Sidecar discovery (Python-compatible): `{name}.lyrics.txt`/`{name}.txt`
   (lyrics), `{name}.json` (caption/bpm/keyscale/timesignature/language/
   is_instrumental), `{name}.caption.txt`. Only `.wav`/`.mp3` audio is
   decodable today (what `src/audio-io.h` implements); `.flac`/`.ogg`/
   `.opus` files are reported and skipped.
2. Missing caption → **understand pipeline** (`pipeline-understand`) fills
   caption, lyrics, bpm, keyscale, language. This is the "Auto Label" step,
   native, and does not overwrite fields the sidecars already provided.
   Duration always comes from the real decoded audio length, never from
   the LM (matching the dataset builder's own "duration is auto-read, not
   LM-guessed" treatment).
3. Encode (all existing inference modules, reused as-is — see the
   Decisions log for the two conditioning conventions this deliberately
   does NOT copy from the Python trainer): audio → 48kHz stereo → tiled
   VAE encode → `target_latents`; `src/dit-prompt.h` builds the SFT prompt
   and the language-wrapped lyric prompt (byte-identical to what
   ace-synth builds for the same inputs) → BPE → Qwen3 encoder (text) /
   embed_tokens (lyrics); CondEncoder(text, lyric, timbre=one silence_latent
   frame) → `encoder_hidden_states`; `context_latents` =
   concat(silence_latent[:T], ones); T is rounded up to a `patch_size`
   multiple (silence-padded) exactly like `ops_resolve_T` does for
   `--src-audio` at inference time.
4. Write one GGUF per sample (`{name}.gguf`) via
   `dit_train_sample_write()` — same 3-tensor format `fit` already
   consumes (`target_latents`, `context_latents`, `encoder_hidden`, each
   unpadded — beyond the patch_size rounding above — at that sample's
   exact length; see the decision below on why there's no separate
   attention_mask tensor).

GGUF as tensor cache = zero new formats, existing reader infra, easy to
inspect with existing tooling.

### fit stage (implemented, Phase 3)

Implemented in `src/dit-train.h` / validated by `tests/test-lora-train-smoke.cpp`
(Phase 2). Design confirmed by that work:

1. Load DiT from GGUF via `dit_ggml_load_train()` (F32 everywhere, always
   the all-separate QKV/gate-up path, `use_flash_attn=false`). Frozen —
   never touched by the optimizer.
2. Create LoRA tensors A (`ggml ne=(in,rank)`, N(0, 1/sqrt(in))) and B
   (`ggml ne=(rank,out)`, zero-init — standard LoRA convention, adapter
   starts as a no-op) in F32 for every self_attn/cross_attn q/k/v/o_proj at
   every layer, plus an AdamW (m, v) pair per tensor. All `ggml_set_param`'d,
   all held in one persistent `WeightCtx`-backed backend buffer alongside
   the frozen backbone's buffer, allocated once for the whole run.
3. Per step, build a **fresh** `ggml_context` (T/enc_S vary per sample) with
   the DiT forward graph (now takes optional `lora_layers`/`want_grads`
   params, additive-only — inference call sites pass neither and are
   byte-identical), a flow-matching MSE loss node, `ggml_build_backward_expand`,
   then one `ggml_opt_step_adamw` node per LoRA tensor built by hand
   (bias-corrected LR terms computed host-side into a shared `adamw_params`
   tensor, matching `ggml-opt.cpp`'s own formula).
4. Checkpoints: `adapter_model.safetensors` (PEFT key naming
   `base_model.model.layers.N.{self_attn,cross_attn}.{q,k,v,o}_proj.lora_{A,B}.weight`)
   + `adapter_config.json` (r, lora_alpha, target_modules), written via
   `src/safetensors-write.h`. Loads unmodified through `src/adapter-merge.h`
   → instantly usable in the WebUI (already proven in Phase 1).
5. Optional `--val-split`: held-out samples evaluated per epoch (forward
   only, no backward/opt nodes), best checkpoint tracked. Not yet built —
   Phase 3.

**Why not `ggml-opt`'s high-level dataset/epoch API**: it keys its persistent
AdamW momentum tensors by *node position* in a graph built once
(`ggml-opt.cpp:462`, `opt_ctx->grad_m[i]` indexed by the forward graph's
node array index `i`), on the assumption a "dynamic" graph's *topology*
(node count/order) stays identical across evals even as the data changes.
Our forward graph's node count and order shift with T (variable song
length), so reusing that path would silently associate the wrong momentum
tensor with the wrong parameter as soon as two samples had different
shapes. Fix: track our own `m`/`v` per LoRA tensor *by pointer identity* (a
plain field in our own `DiTTrainLayerMomentum` struct) and call
`ggml_opt_step_adamw` ourselves — same underlying op, safe under variable
shapes. Grad clipping (Python's `max_grad_norm=1.0`) is still not
implemented (Phase 3 didn't need it for the synthetic-data validation run;
revisit before training on real songs where outlier gradients are more
plausible).

**Phase 3 split the single training step in two** (`dit_train_forward_backward`
+ `dit_train_optimizer_step`) so `--grad-accum` can actually accumulate
across several samples before spending an AdamW step. Since each
micro-batch's backward pass lives in its own fresh, differently-shaped
`ggml_context`, there's no single persistent graph to add gradients into
across calls -- so accumulation happens on the *host*: each
`forward_backward` call reads back that step's freshly computed gradient
per LoRA tensor and sums it into a plain `std::vector<float>` accumulator
(`DiTTrainProjGradAccum`, parallel to the LoRA structure). `optimizer_step`
then builds one small graph (no forward pass, just one
`GGML_OP_OPT_STEP_ADAMW` view-node per tensor), uploads the accumulator
divided by the micro-batch count as that op's `grad` input, runs it, and
zeroes the accumulator. Cheap regardless of `grad_accum`: the graph is
`O(n_layers * 8)` nodes, not tied to sequence length at all.

### Phase plan

- [x] **Phase 0 — study & feasibility** (this document)
- [x] **Phase 1 — safetensors writer + LoRA checkpoint round-trip**
  `src/safetensors-write.h` (minimal writer, counterpart to the existing
  reader) + `tests/test-lora-roundtrip.cpp`: generates synthetic LoRA A/B for
  every self_attn/cross_attn q/k/v/o_proj tensor in a real DiT GGUF, writes a
  PEFT adapter directory, runs it through the production `adapter_merge()`
  path on the CPU backend, and checks the merged weights against a
  host-computed reference. Verified against `acestep-v15-turbo-Q8_0.gguf`:
  192/192 tensors (24 layers x 8 projections) merged, min cosine similarity
  0.999982. Output format is confirmed byte-compatible with what
  `src/adapter-merge.h` already loads for inference.
- [x] **Phase 2 — training graph + backward smoke test**
  `src/dit-train.h` (LoRA injection in `src/dit-graph.h`, additive-only
  trailing params; F32 training loader in `src/dit.h`) +
  `tests/test-lora-train-smoke.cpp`. Confirmed on the real
  `acestep-v15-turbo-Q8_0.gguf`, full 24 layers, no CPU fallback needed
  (CUDA only): backward builds and computes without any unsupported-op
  abort, frozen backbone stays bit-identical, LoRA B moves off its zero
  init, and a fixed single-sample overfit drives loss from 1.78 → 0.000047
  over 150 steps (clean monotonic convergence). Two real bugs found and
  fixed along the way — see Decisions log. Not yet measured: VRAM at
  production T (~2000+ latent frames); this test uses small synthetic
  T/enc_S (32/8) to iterate quickly. CPU backend path not yet exercised
  (only CUDA available on the dev machine used for this phase).
- [x] **Phase 3 — `ace-train fit`**
  `tools/ace-train.cpp` (subcommand dispatch: `fit` implemented, `prepare`
  reports "not yet, see Phase 4" rather than guessing) +
  `src/dit-train-data.h` (per-sample tensor cache: 3 unpadded GGUF tensors —
  `target_latents`, `context_latents`, `encoder_hidden` — read/written via
  the existing `gguf.h` reader/writer, no new format). `dit-train.h` gained
  `dit_train_forward_backward`/`dit_train_optimizer_step` (split for grad
  accumulation, see Design above), `dit_train_eval` (forward-only, for
  validation), and `dit_train_save_checkpoint` (PEFT directory via
  `safetensors-write.h`). The fit loop: shuffles + splits samples
  (`--val-split`), resamples noise + a discrete turbo timestep per sample
  per pass (matching the Python trainer), groups samples into
  `--grad-accum`-sized micro-batches (leftover partial batch at epoch end
  handled correctly), applies the warmup+cosine LR schedule per optimizer
  step, evaluates on the held-out split every epoch (best-loss checkpoint),
  and saves periodic + final checkpoints.
  **Validated end to end** with `tests/gen-train-samples.cpp` (synthetic
  sample generator, since Phase 4 doesn't exist yet) against the real
  24-layer `acestep-v15-turbo-Q8_0.gguf`: full fit run completes (shuffle,
  accumulate, schedule, checkpoint, val-eval) and the resulting
  `final/adapter_model.safetensors` loads through the *actual production*
  `ace-synth --adapters` path — **192/192 tensors merged, 0 skipped** — and
  renders a track. Also exercised uneven `--grad-accum` (sample count not a
  multiple of the accumulation window) with no error. Two bugs found and
  fixed during this phase (both in `dit_train_sample_write`/the ctx-memory
  budget it uses, and the usage-string program-name bug in `ace-train.cpp`)
  — see Decisions log.
- [x] **Phase 4 — `ace-train prepare`**
  `src/dit-prepare.h` (sidecar scan, auto-label via the understand pipeline,
  audio -> VAE/text/lyric/cond encode) + `src/dit-prompt.h` (the DiT
  text/lyric prompt builder, extracted out of `pipeline-synth-ops.cpp` so
  inference and dataset prep can never drift apart — see Decisions log for
  why this matters). Sidecar schema matches the Python trainer exactly
  (`{name}.json`, `{name}.lyrics.txt`/`.txt`, `{name}.caption.txt`); audio
  decoding is limited to .wav/.mp3 (what `src/audio-io.h` implements today —
  .flac/.ogg/.opus sidecars are reported and skipped, not silently dropped).
  **Validated end to end** against real music files (a mix of tracks with
  full sidecars and one with none, to exercise auto-labeling) from a local
  `/tmp/music` sample library: `ace-train prepare` encoded all samples with
  no failures (including successfully auto-labeling the sidecar-less file
  via the LM, hallucinated caption content and all — a known, expected LM
  behavior, not a pipeline bug), `ace-train fit` trained on the resulting
  tensor cache with a real train/val split, and the resulting checkpoint
  loaded through the *production* `ace-synth --adapters` path — 192/192
  tensors merged, 0 skipped — for both a single-sample and a 3-sample real
  batch. One real bug found and fixed: VAE-encoded `T` isn't guaranteed to
  be a multiple of `patch_size` (a 4.47s clip encoded to T=111, odd); the
  DiT's patchify reshape requires `T % patch_size == 0` and aborted
  (`ggml_reshape_3d` assert) on the unpadded sample. Fixed by rounding T up
  and padding the tail with `silence_latent` frames, exactly mirroring how
  `ops_resolve_T` already does this for `--src-audio` at inference time
  (`src/pipeline-synth-ops.cpp:330`).
- [~] **Phase 5 — end-to-end validation** (in progress)
  Train a small-style LoRA on 10–20 songs; A/B the adapter in ace-synth /
  WebUI vs base model; compare loss curves against the Python trainer on the
  same dataset (both start from the same BF16 turbo DiT).
  Step 1 (single-song sanity check) done: fit 300 epochs on one real
  70s breakbeat track (`grad-accum=1`, no val-split — overfitting one real
  song on purpose, to isolate "does the pipeline learn from real audio at
  all" from "does it generalize across a diverse set"). EMA loss dropped
  1.29 → 0.70 over the run (noisy, as expected — real training resamples
  noise + a discrete timestep every pass, unlike Phase 2's fixed-target
  overfit test, so this is not a clean monotone curve). Real GPU throughput
  at production-scale T (~1500 latent frames, 60s audio): ~1s/optimizer
  step uncontended. A/B rendered from the same LM-generated codes/seed
  through both the base model and the trained adapter (scale 1.0);
  confirmed genuinely different output (`md5sum` differs, adapter reports
  192/192 tensors merged) and handed both tracks to the user to judge by
  ear — model can't self-assess audio quality, only mechanical correctness.
  One workflow mistake caught immediately: `ace-synth --adapters <dir>`
  does nothing unless the request JSON itself carries `"adapter": "<name>"`
  — an LM-generated request has no such field by default, so the very
  first "with adapter" render silently ran as base (identical file size,
  caught by comparing hashes). Not a bug in `ace-synth` (this matches
  documented behavior, README's Adapters section: "Select the active
  adapter" is a per-request field, not a directory-presence toggle) —
  purely an operator error worth flagging for anyone scripting A/B
  comparisons the same way.
  A wider 14-song diverse-style run was also started (300 epochs,
  grad-accum=2, val-split 0.15, ~1h50m projected) to test a real multi-song
  dataset, but stopped partway (at epoch ~155/300, EMA ~0.87, best val-loss
  0.826) in favor of the cleaner single-song check first — GPU contention
  between the two concurrent jobs was part of the reasoning. Revisit with
  a deliberately curated set once the single-song result is confirmed
  good by ear.
  Step 2 (second single-song check, `sample-A.mp3`, no
  sidecars) surfaced a real process failure, not a mechanism bug: auto-label
  hallucinated the caption as "dreamy, atmospheric... chillwave or ambient
  electronic music" for a track the user describes as an epic house
  track with powerful distorted bass and heavy sidechain pumping — the
  LM got the genre completely backwards. That wrong caption then drove
  *both* training's text conditioning (paired against the real house-track
  audio in `target_latents` — a contradictory training signal) *and* the
  A/B generation (both renders asked for chillwave, so of course both
  sounded like chillwave, regardless of what the LoRA learned from the
  audio side). This is the single-file re-run of the exact hallucination
  risk already noted for an earlier test clip in Phase 4's
  progress log, except that time nobody used the bad caption for anything
  downstream, so it went unnoticed. **Lesson**: never treat an auto-labeled
  caption as ground truth without a sanity check — a human glance at the
  caption before training (or before using it for A/B generation) would
  have caught this immediately. Fixed by writing a corrected `.json`
  sidecar by hand and re-running `prepare` (confirmed via the log: no
  "Auto-labeled" line this time, meaning the sidecar caption was used
  directly) and `fit`. Also spot-checked the first run's trained LoRA
  delta magnitude directly from the safetensors (layer 0 self_attn.q_proj:
  `lora_A` norm 2.87 / max 0.098, `lora_B` norm 0.148 / max 0.0055) — small
  but clearly non-degenerate, consistent with "the adapter learned
  *something* from 300 steps, it was just conditioned on a caption that
  contradicted the audio it was aligned with." Re-run with the corrected
  caption in progress.
  Step 3: user raised a real methodology gap while the corrected re-run was
  in progress — a *long descriptive caption* is not how you reliably
  invoke a trained concept; you want a short, consistent **trigger word**
  every training sample shares, so the adapter learns one dependable hook
  instead of the model having to re-derive "the training distribution"
  from a differently-worded caption every time. Checked: acestep-repo
  already has exactly this (`AudioSample.custom_tag` +
  `DatasetMetadata.tag_position`, `get_full_caption()` in `models.py`) —
  our `ace-train prepare` had no equivalent at all until now. Added
  `--trigger-word` / `--tag-position` (prepend/append/replace, prepend
  default, matching Python) to `ace-train prepare`; implemented as
  `dit_prepare_apply_tag()` in `src/dit-prepare.h`, byte-for-byte the same
  string-building logic as `get_full_caption()`. In `replace` mode,
  auto-labeling is skipped entirely (the trigger word is the whole prompt
  regardless of caption, so there's nothing for the LM to contribute and
  no hallucination risk to import). Re-running the sample-A training
  with `--trigger-word acetrig01 --tag-position prepend` layered on top of
  the corrected caption, to test whether the short trigger word alone (no
  long description needed) reliably invokes the learned qualities at
  inference — that's the actual open question, not yet answered.
  Trained (EMA 1.40 → 0.91). Testing it surfaced a *third* real
  operational gap, this time in how `ace-lm` is invoked, not in
  `ace-train`: sending `{"caption": "acetrig01", "lyrics": ""}` through
  `ace-lm` triggers Phase 1 CoT (per `docs/ARCHITECTURE.md`'s "Caption
  only" mode), which **freely rewrites** an unrecognized short caption
  into something else entirely — one run turned `"acetrig01"` into "A
  clean, nylon-string acoustic guitar plays a gentle, arpeggiated chord
  progression..." before it ever reached the DiT's text encoder. The
  trigger word never made it into the conditioning at all; that render
  was meaningless as a test. Fix: set `"lyrics": "[Instrumental]"` (so
  Phase 1 isn't in free-caption-generation mode) and
  `"use_cot_caption": false` (so even the CoT pass that still runs to fill
  missing bpm/keyscale/etc. leaves the caption text untouched — verified
  the resulting `request0.json` has `"caption": "acetrig01"` byte for
  byte). **Anyone using a trigger word through the full ace-lm → ace-synth
  pipeline needs `use_cot_caption: false`**, or the trigger word is at the
  mercy of the LM's caption-enrichment step and may not survive to reach
  the model at all. Re-ran the A/B with this fix.
  User's verdict on that A/B: base and adapter were "basically the same"
  and neither resembled the reference track at all — a real signal that
  something is fundamentally off, not just "needs more training." Went
  back to the Python reference (`acestep-repo`) specifically for how it
  applies a trained LoRA at inference and validates it, rather than keep
  guessing. Findings (see full research notes; key facts below):
  - **The Python reference never merges LoRA into a quantized checkpoint,
    and explicitly refuses to.** LoRA is applied as a *dynamic* PEFT wrapper
    (`peft.PeftModel.from_pretrained`, `.../lora/lifecycle.py:191-272`) kept
    separate from the base weights at inference; quantized models are
    rejected outright with `"❌ LoRA loading is not supported on quantized
    models"` (same file, ~line 200). Training itself also refuses
    torchao-quantized decoders. A `merge_and_unload()` path exists
    (`training/lora_utils.py`) but is an export-only utility, not part of
    the generation path. There is no "dequantize base, add delta,
    requantize" workflow anywhere in that codebase — acestep.cpp's
    adapter-merge path (necessary for a static-merge, zero-runtime-overhead
    design, and normally fine for a *well-trained* adapter) is the only
    place doing this, and it's now a live suspect.
  - **LoRA scale**: their UI defaults to and clamps to 1.0 ("full
    strength") — confirms `adapter_scale=1.0` was the right value to test
    with; this isn't a scale-convention mismatch.
  - **Epoch guidance**: their own docs recommend 200-500 epochs for
    1-10 songs (`docs/sidestep/Training Guide.md`), 800 for a 10-20 song
    style set. Our single-song runs used 300 *steps* (grad-accum=1, so
    steps==epochs here) — in range for their low end, not obviously too
    short, though "epoch" in their multi-song runs means one full pass
    with grad-accum=4, i.e. more actual gradient updates per epoch than
    our single-sample runs get per step. No source found stating a
    specific "minimum epochs before any audible effect" threshold.
  - Wrote a quantitative diagnostic (`tests/diag-quant-erasure.cpp`, ad hoc,
    not wired into the build) that runs the *real* trained checkpoint
    through the *real* `adapter_merge()` against the *real* Q8_0 DiT and
    compares dequantized merged vs. original weights directly. Result on 5
    real projections: the merge does change real bytes (not a no-op:
    63-96% of bytes came out byte-identical, meaning 4-37% changed), but
    the changes are tiny — **mean |delta| / mean |base weight| ranged
    0.12% to 1.05% across layers 0/12/23**. That is a very small
    perturbation in absolute terms, on top of which quantization rounds
    away part of an already-small signal. Two compounding causes, not one:
    (1) 300 steps at lr=1e-4 on a single 60s clip is producing a genuinely
    faint weight shift — likely underpowered regardless of quantization;
    (2) merging into Q8_0 (which the Python reference never does at all)
    coarsens that already-faint signal further.
  - Ran one more diagnostic: `adapter_scale=20.0` (20x the Python
    reference's own max) on the *same* trained checkpoint, no retraining,
    to check whether the learned direction is real but just too weak at
    scale=1, or genuinely absent. Rendered and handed to the user alongside
    the corrected-caption/trigger-word A/B for a perceptual verdict on all
    three.
  - **Recommended next steps** (not yet acted on, pending the scale=20
    listening result): (a) train substantially harder before drawing
    conclusions — more steps and/or a higher learning rate, since Python's
    own epoch guidance implies far more gradient updates than our
    single-song runs have used so far; (b) treat quantized-model LoRA
    merging as a real fidelity risk the way the Python reference does —
    either warn/require an unquantized (BF16) DiT for `--adapters` the way
    they refuse quantized models outright, or keep it but document the
    expected precision loss; (c) don't judge "does the mechanism work" from
    a single 300-step/rank-8/one-song run again — use a stronger run as the
    baseline for any future A/B.
  User pushed back on the amplification line of investigation (scale=3/5/20
  all reported as "just noise") and gave a crucial data point: **their own
  past Python-pipeline experiment training on one song did work** — not
  stylistically, but as a near-literal copy, confirming single-song
  "memorization" is achievable on this architecture when done right. That
  ruled out "single-song reproduction is an unrealistic ask" as an
  explanation and pointed the investigation back at *our* pipeline
  specifically diverging from Python's, not at the technique itself.
  Re-read `docs/en/LoRA_Training_Tutorial.md` in full (previously only
  summarized) and cross-checked a second, independent Python reference
  the user pointed at,
  [ComfyUI-FL-AceStep-Training](https://github.com/filliptm/ComfyUI-FL-AceStep-Training)
  (cloned to `../ComfyUI-FL-AceStep-Training`), for an exact parameter and
  pipeline comparison:
  - **Hyperparameters match exactly** across all three implementations
    (ours, acestep-repo, ComfyUI-FL): rank=8, alpha=16, target_modules
    q/k/v/o_proj, lr=1e-4, AdamW + warmup/cosine, turbo discrete
    shift=3.0 8-timestep schedule, MSE flow-matching loss
    (`xt = t*x1+(1-t)*x0`, target `x1-x0`). No hyperparameter bug found.
  - **Training duration is the one place we're clearly out of range.**
    ComfyUI-FL's own `max_epochs` slider goes up to 10,000 (default 100);
    the tutorial's own disclaimer demo used 500 epochs — on a 13-song
    album, not one song, but still far more optimizer steps in aggregate
    than our 300. Nothing we tested so far has gone past low hundreds of
    steps on a single sample.
  - **Two more real divergences found**, both in dataset encoding
    (`src/dit-prepare.h`), both traced back to an earlier deliberate
    choice to match ace-synth's *inference* convention instead of the
    Python *trainer's* convention:
    1. Lyrics: we were wrapping as
       `"# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>"` before
       tokenizing (matches ace-synth inference). Both acestep-repo
       (`preprocess_lyrics.py`) and ComfyUI-FL
       (`comfy_wrappers.py::embed_tokens`) tokenize **raw, completely
       unwrapped** lyrics text, `embed_tokens` only, max_length=512 — no
       wrapper of any kind. Two independent references agree; ours didn't
       match either.
    2. CondEncoder timbre placeholder: we were feeding one frame of the
       DiT's real `silence_latent` (matches ace-synth inference).
       acestep-repo's `run_encoder` and ComfyUI-FL's
       `_get_refer_audio_tensors` both use a literal
       `torch.zeros(1, 1, 64)` dummy for training. (The **context_latents**
       src channels — silence + mask=1 for text2music — were never in
       question: both Python references and ace-synth all agree those use
       the real silence_latent; only the *timbre* placeholder differed.)
  - Given the user now explicitly wants exact Python-pipeline replication
    rather than ace-synth-inference-consistency, **reverted both**:
    lyrics are now tokenized raw/unwrapped, and the timbre placeholder is
    now a zero vector, matching both Python references byte-for-byte. This
    supersedes the earlier Phase 4/5 decisions to do the opposite; if a
    production LoRA later needs ace-synth-inference-matched conditioning
    instead, that's a deliberate follow-up, not the default.
  - Re-ran `prepare` (confirmed via log) and launched a much longer `fit`
    run — 3000 steps (vs. 300 before) on the corrected encoding, closer to
    the ecosystem's actual typical scale — to test the "just needed way
    more training, on the right encoding" hypothesis properly. In progress;
    see Progress log for the outcome once it lands.
- [ ] **Phase 6 — polish / stretch**
  Server endpoints (`/training/*`) + WebUI tab, SSE progress; LoRA dropout
  via philox mask; gradient checkpointing; BF16 out_prod fork patch;
  LoKr.

## Decisions log

- 2026-07-23: train turbo (shift=3, 8 discrete timesteps) first — same as
  Python default path. Base/SFT continuous-timestep training later if wanted.
- 2026-07-23: GGUF for tensor cache, PEFT safetensors dir for output.
- 2026-07-23: training always uses no-FA + split-swiglu graph paths
  (backward coverage); inference paths untouched.
- 2026-07-23: F32 dequant of DiT weights during training (CUDA out_prod
  constraint); revisit with cast-nodes or fork patch if VRAM-bound.
- 2026-07-23: hand-roll the training step (`ggml_build_backward_expand` +
  manual per-tensor `ggml_opt_step_adamw`) instead of `ggml-opt`'s
  high-level dataset/epoch API — that API's momentum persistence is keyed
  by graph node position, which is unsafe when T varies per sample (see
  fit-stage design above). Our own `DiTTrainLayerMomentum` struct keys
  momentum by parameter identity instead.
- 2026-07-23: two bugs found by the Phase 2 smoke test, both fixed:
  (1) `dit_ggml_load_train()` must zero the whole `DiTGGML` struct up
  front (`*m = {}`) — it never assigns the fusion-candidate fields
  (`sa_qkv`, `sa_qk`, `ca_qkv`, `ca_kv`, `gate_up`) since training always
  takes the all-separate path, so on an un-zeroed struct they hold
  indeterminate garbage instead of nullptr, occasionally taking the wrong
  (fused) branch in `dit-graph.h` with a mismatched tensor pointer. Bit
  every layer with nonzero probability, so it passed at 2 layers and
  crashed at 24 (`ggml_can_mul_mat` assert in `mul_mat`).
  (2) A freshly built backward graph's loss-gradient accumulator is not
  seeded to 1.0 by default — `ggml_graph_reset()` normally does that, but
  it *also* zeroes every `GGML_OP_OPT_STEP_ADAMW` node's momenta
  (`src[2]`/`src[3]`), which alias our persistent m/v tensors, so calling
  it every step would erase Adam's cross-step history. Fix: seed only the
  loss's own grad accumulator directly
  (`ggml_backend_tensor_set(ggml_graph_get_grad_acc(gf, loss), &onef, ...)`)
  instead of calling the blanket reset. Symptom before the fix: graph
  computed and the optimizer step "ran" with no crash, but every gradient
  was exactly zero, so only weight decay moved the parameters.
- 2026-07-23: gf_load_tensor_f32 (src/gguf-weights.h) extended to dequant
  *any* type via the generic `ggml_type_traits->to_float` (previously only
  BF16/F16; quantized types silently fell back to loading native, which
  would have defeated the F32-everywhere training requirement on a
  Q8_0/K-quant checkpoint). Purely additive: existing callers only ever
  passed BF16/F16/F32 tensors, so inference is unaffected (re-verified with
  a full ace-lm/ace-synth round trip after the change).
- 2026-07-23: sample tensor cache (`src/dit-train-data.h`) drops
  `attention_mask`/`encoder_attention_mask` from the design doc's original
  5-tensor plan. Each sample is trained at its own exact, unpadded length
  (N=1 per graph build, never batched), so an all-valid mask is already
  correct — `dit-train.h` already builds one internally. Only revisit this
  if/when samples are ever batched to N>1 in one graph.
- 2026-07-23: Phase 3 gradient accumulation happens on the **host**, not in
  a persistent graph. Each micro-batch's backward pass lives in its own
  fresh, differently-shaped `ggml_context` (T/enc_S vary per sample), so
  there's no single graph node to accumulate into across calls the way a
  fixed-shape training loop could. `dit_train_forward_backward` reads back
  each LoRA tensor's freshly computed gradient and sums it into a plain
  host `std::vector<float>`; `dit_train_optimizer_step` uploads the
  averaged sum as a small forward-only graph's `grad` input. Simple, and
  cheap regardless of `grad_accum` since the tensors are LoRA-sized
  (kilobytes), not backbone-sized.
- 2026-07-23: two bugs found by Phase 3's end-to-end run, both fixed:
  (1) `dit_train_sample_write()` (`src/dit-train-data.h`) allocated its
  `ggml_context` with `no_alloc=false` but only budgeted tensor *struct*
  overhead, not the tensor *data* itself — crashed
  (`ggml_new_object: not enough space`) on the first real sample larger
  than a few hundred bytes. Fixed by adding the actual data byte count
  (`64*T + 128*T + H_enc*enc_S` floats) to the context size.
  (2) `tools/ace-train.cpp`'s `run_fit()` used the post-subcommand-shifted
  `argv[0]` (i.e. the first *flag*, not the program name) in its own usage
  string. Fixed by passing the real `argv[0]` through explicitly from
  `main()`.
- 2026-07-23: **Phase 4 conditioning-convention decisions.** While
  implementing `dit_prepare_encode_sample`, found that
  `acestep-repo/acestep/training/dataset_builder_modules` (the Python LoRA
  trainer's own dataset prep) uses different conditioning conventions than
  the original ACE-Step inference pipeline in two places:
  - **Lyrics**: the Python trainer's `preprocess_lyrics.py` tokenizes raw,
    unwrapped lyrics text. Both the original Python inference pipeline
    (`acestep-repo/.../prompt_utils.py::_format_lyrics`) and ace-synth
    (`build_prompt_strings`) wrap lyrics as
    `"# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>"` first.
  - **Timbre placeholder**: the Python trainer passes `zeros(1, 1, 64)`
    to the CondEncoder's timbre branch when there's no reference audio
    (`preprocess_encoder.py::run_encoder`). ace-synth instead passes one
    frame of the DiT's own `silence_latent` (`ctx->meta->silence_full`,
    `src/pipeline-synth-ops.cpp:369/381/390`) — a real learned VAE
    encoding of silence, not literal zeros.
  User confirmed (asked directly, see conversation): match ace-synth's own
  conventions for both, not the Python trainer's. Rationale, stated by the
  user and consistent throughout: a LoRA trained here is used *through
  ace-synth*, so it must see exactly the conditioning distribution
  ace-synth actually produces at inference time; bit-for-bit fidelity to
  the community trainer's dataset code is secondary to that. Implemented
  in `src/dit-prompt.h` (extracted out of `pipeline-synth-ops.cpp` as a
  shared function — `dit_build_prompt_strings` — specifically so inference
  and `ace-train prepare` can never drift apart on this again) and
  `dit_prepare_encode_sample`'s CondEncoder call (`meta->silence_full`,
  1 frame, `S_ref=1`).
  The **context_latents src channels** (text2music: silence + mask=1) do
  *not* have this conflict — both the Python trainer
  (`preprocess_context.py::build_context_latents`) and ace-synth already
  use the real `silence_latent`, not zeros. (Earlier Phase 2/3 test code —
  `tests/gen-train-samples.cpp`, `tests/test-lora-train-smoke.cpp` — used
  literal zeros there, which was fine for those phases' mechanism-only
  smoke tests but is *not* what `dit_prepare_encode_sample` does for real
  data; it correctly uses `silence_full`.)
- 2026-07-23: found and fixed a real bug via the real-audio end-to-end
  test: VAE-encoded `T` (latent frame count) is not guaranteed to be a
  multiple of the DiT's `patch_size` — a 4.47s clip encoded to T=111
  (odd). `dit_ggml_build_graph`'s patchify reshape requires
  `T % patch_size == 0` and aborted with a `ggml_reshape_3d` assertion on
  the unpadded sample. ace-synth already handles this for `--src-audio`
  inputs by rounding `T` up to the next multiple of `patch_size`
  (`ops_resolve_T`, `src/pipeline-synth-ops.cpp:330`); `dit_prepare_encode_sample`
  now does the same, padding both `target_latents` and (implicitly, via
  the later loop) `context_latents` with `silence_latent` frames for the
  tail.

## Progress log

- 2026-07-23: Studied Python trainer + tutorial + preprocessing modules.
  Audited GGML fork autograd/opt coverage against the DiT graph op
  inventory — **no fork changes required to start**. Branch `training`
  created, plan written.
- 2026-07-23: Phase 1 complete. safetensors writer + round-trip test added
  and passing against the real turbo DiT GGUF (see Phase 1 above).
- 2026-07-23: Phase 2 complete. LoRA training graph (forward + backward +
  AdamW) validated end-to-end on the full 24-layer production DiT; found
  and fixed two real bugs (garbage fusion-pointer fields, unseeded loss
  gradient) along the way — see Decisions log.
- 2026-07-23: Phase 3 complete. `ace-train fit` CLI + per-sample tensor
  cache (`src/dit-train-data.h`) + gradient-accumulation split in
  `dit-train.h` (forward_backward/optimizer_step/eval/save_checkpoint).
  Validated end to end on the real 24-layer turbo DiT with a synthetic
  sample generator (`tests/gen-train-samples.cpp`, since Phase 4 doesn't
  exist yet): full fit run (shuffle, grad-accum incl. uneven leftover
  batches, LR schedule, val-split, checkpointing) completes cleanly, and
  the resulting checkpoint loads through the *actual* `ace-synth
  --adapters` production path with 192/192 tensors merged, 0 skipped, and
  renders a track. Two small bugs found and fixed — see Decisions log.
  Next up: Phase 4, `ace-train prepare` (real dataset decoration via the
  understand pipeline + VAE/text/cond encoding, replacing the synthetic
  generator).
- 2026-07-23: Phase 4 complete. `ace-train prepare` implemented
  (`src/dit-prepare.h`) and validated against real music files (mixed
  sidecar-complete and sidecar-less, from a local `/tmp/music` library):
  sidecar scan + auto-label + VAE/text/lyric/cond encode all working, one
  real bug found and fixed (VAE-encoded T not always a multiple of
  patch_size — see Decisions log), and the full `prepare` -> `fit` ->
  `ace-synth --adapters` chain validated end to end on real audio
  (192/192 tensors merged, 0 skipped, both single-sample and 3-sample
  batches). Two deliberate conditioning-convention departures from the
  Python trainer's dataset builder (lyric wrapper, timbre placeholder),
  confirmed with the user and documented in the Decisions log, along with
  the shared `src/dit-prompt.h` extraction that keeps inference and
  training conditioning from ever drifting apart. Next up: Phase 5,
  end-to-end validation (train a real style LoRA on 10-20 songs, A/B
  against the base model, compare against the Python trainer on the same
  dataset).

## Verification checklist (running)

- [x] Saved safetensors loads via adapter-merge with correct alpha/rank scaling
      (`tests/test-lora-roundtrip.cpp`, 192/192 tensors, cossim 0.999982)
- [x] Backward graph builds without abort for full 24-layer DiT (CUDA;
      CPU backend not yet exercised, no CPU-only dev machine used so far)
- [x] LoRA A/B grads nonzero after 1 step; frozen weights bit-identical
      (`tests/test-lora-train-smoke.cpp`)
- [x] Overfit single sample: loss → ~0 (1.78 → 0.000047 over 150 steps,
      full 24-layer model)
- [x] `ace-train fit` runs a full loop (shuffle, grad-accum incl. uneven
      leftover batches, LR warmup+cosine schedule, val-split eval,
      periodic + best + final checkpoints) against real sample tensor
      files with no crash or FATAL
- [x] `ace-train fit`'s checkpoint output loads through the *production*
      `ace-synth --adapters` path (192/192 tensors merged, 0 skipped) and
      renders audio
- [x] `ace-train prepare` encodes real audio (.mp3) end to end: sidecar
      scan, auto-label via understand when captions are missing, VAE/text/
      lyric/cond encode, T-padding to a patch_size multiple, all with no
      crash or FATAL, on a mixed batch of sidecar-complete and
      sidecar-less real files
- [x] `ace-train prepare` -> `fit` -> `ace-synth --adapters` full chain
      validated on real audio (192/192 tensors merged, 0 skipped, single-
      sample and 3-sample batches)
- [ ] Same backward graph verified on CPU backend specifically
- [ ] soft_max_ext backward correct with a *non-trivial* attention mask
      (real prepared samples still use all-valid masks: each sample is
      encoded and trained at its own exact — now patch-padded — length,
      never batched with others, so there's no padding-driven masking to
      exercise yet)
- [ ] VRAM measured at production-length songs (full 3-6 minute tracks;
      testing so far used short clips for iteration speed)
- [ ] prepare-stage tensors match Python preprocessed .pt (cossim) on the
      same real sample (would need a working acestep-megalora preprocess
      run to compare against — not yet done)
- [~] Trained LoRA audibly shifts style in ace-synth output: single-song
      (one real track, 300 epochs) A/B rendered and handed to the user for
      listening judgment — mechanically confirmed different output,
      perceptual verdict pending. Multi-song style dataset not yet done.

## Perf: per-step GPU idle gaps in `ace-train fit` (fixed)

User noticed `nvidia-smi dmon` showing the GPU alternating ~96% / ~0%
utilization in a multi-second cycle during a long single-song run (3000
steps, grad-accum=1), rather than staying busy. Root cause, found in
`src/dit-train.h`: `dit_train_forward_backward`, `dit_train_eval`, and
`dit_train_optimizer_step` each built a fresh `StaticGraph` (i.e.
`ggml_gallocr_new` -> `alloc_graph` -> `ggml_gallocr_free`) on *every
call*, i.e. up to 3x per training step. `ggml_gallocr_free` triggers a
real `ggml_backend_buffer_free` (cudaFree) and the next `ggml_gallocr_new`
+ `alloc_graph` forces a fresh reservation (cudaMalloc) since there's no
previous allocation to compare against — two full alloc/free round trips
per step, serializing the CUDA driver and showing up as idle GPU gaps.

Fix: `ggml_gallocr` is designed to be created once and reused across many
`alloc_graph` calls on different (but same-shape) graphs —
`ggml_gallocr_alloc_graph` only reallocates when the incoming graph's
node/leaf count or tensor shapes actually differ from what's already
reserved (`ggml_gallocr_needs_realloc` in `ggml-alloc.c`); otherwise it's
just a cheap bookkeeping reset. Added three persistent `ggml_gallocr_t`
handles to `DiTTrain` (`fwd_galloc`, `eval_galloc`, `opt_galloc`), created
once in `dit_train_init`, freed once in `dit_train_free`, and reused
directly in place of the old per-call `StaticGraph`. Since every step
trains the same sample (same T) and the optimizer-step graph's topology
never changes, this eliminates essentially all per-step
cudaMalloc/cudaFree calls after the first step.

Verified:
- `tests/test-lora-train-smoke.cpp` still PASSes (loss curve identical:
  1.776807 -> 0.753187 over 60 steps, same as before the change) —
  confirms the reuse doesn't change training semantics.
- `tests/test-lora-roundtrip.cpp` still PASSes (192/192 tensors, cossim
  0.999982) — confirms checkpoint format/content unaffected.
- Steady-state throughput on the real single-song dataset went from
  ~0.49-0.53 s/step to ~0.31 s/step (~40-45% faster wall-clock).
- `nvidia-smi dmon` during a fresh run now shows sustained 70-96% SM
  utilization with no idle gaps, vs. the previous 96%/0% alternation.

This was pure infra overhead; it doesn't change any of the perceptual/
under-training investigation above. `static-graph.h`'s `StaticGraph`
helper itself is untouched and still used as-is by the (already one-shot,
non-looping) inference call sites in `src/qwen3-lm.h` and
`src/dit-sampler.h` — out of scope here.

## Metadata conditioning (bpm/keyscale/timesignature)

Training encodes these directly into the DiT's Metas text block
(`dit_build_prompt_strings` in `dit-prompt.h`; `bpm<=0` renders as
literal `"N/A"`). They must be **non-placeholder and identical** at
train and generation time:

- `dit_prepare_encode_sample()` only auto-labels (fills bpm/keyscale/
  timesignature/caption from `ace-understand`) when `label.caption` is
  empty (`dit-prepare.h`, "Auto-label if caption is missing" gate). A
  sidecar with its own caption but `bpm: 0` stays at `0` -- never write
  a caption without also supplying the real bpm/keyscale/timesignature.
- At generation, `ace-lm`'s Phase 1 (`metadata-fsm.h`) gap-fills only
  fields the request left empty. Set `bpm`/`keyscale`/`timesignature`/
  `vocal_language` explicitly in the request to the exact training
  values -- otherwise the LM substitutes its own guess.
- `ace-understand`'s LM-based bpm/key detection is unreliable for
  precise numeric estimation (measured miss: guessed bpm=71/key=C#minor
  vs. confirmed true bpm=126/key=F minor on a full untruncated track) --
  treat as a draft, verify by ear or DSP (see Essentia below), never
  trust blindly.

## Essentia integration: DSP-based bpm/key detection

Vendored [Essentia](https://essentia.upf.edu/) (MTG, AGPL-3.0) as a git
submodule (`vendor/essentia`) for DSP-based tempo/key analysis
(`RhythmExtractor2013`, `KeyExtractor`) to replace the LM's unreliable
numeric-metadata guesses.

**Build**: `waf` (Essentia's own build), wired into CMake via
`add_custom_command`/`add_custom_target` (`essentia_build`) ->
`libessentia.a`, wrapped as an `IMPORTED` target linked **only** into
`ace-train` and `ace-understand` directly -- **not** into `acestep-core`.
`acestep-core` is the static library shared by every binary (`ace-synth`,
`ace-lm`, `ace-server` too); this project is MIT-licensed, and Essentia is
AGPL-3.0, so linking it into the shared library would carry AGPL's
copyleft obligations onto binaries that never call into it. Verified with
`nm -C` on the built binaries, not just by reading the CMake file: zero
Essentia symbols in `ace-synth`/`ace-lm`/`ace-server`, present (as
expected) in `ace-train`/`ace-understand`. Lightweight mode (`--lightweight=fftw,yaml
--build-static`): FFTW3 + libyaml only, no FFmpeg/libsamplerate/TagLib/
Gaia2/TensorFlow (we do our own decode+resample and don't need file I/O
or ML classifiers). Builds on ARM64 (NVIDIA GB10) with
`libeigen3-dev`/`libfftw3-dev`/`libyaml-dev` from apt, ~52s for 294
translation units.

**API**: `src/audio-analysis.h`.
- `audio_analyze_bpm_key_buf()` -- low-level, pre-decoded buffer + rate.
- `audio_analyze_bpm_key_from_file()` -- use this one. Decodes at the
  file's **native** rate, resamples once directly to 44100 Hz
  (`RhythmExtractor2013` hardcodes an internal 44100 Hz assumption).
  Resampling from an already-48kHz-resampled buffer (native->48k->44.1k,
  two lowpass passes) measurably degrades tempo tracking: 126.0 exact
  vs. 144.6 on the same confirmed-bpm=126 track. Always feed it the
  original file path, not a buffer the rest of the pipeline already
  resampled.

**Call sites** (fill gaps only, never override an explicit value):
- `tools/ace-understand.cpp`: overrides `out.bpm`/`out.keyscale` after
  `ace_understand_generate()` returns.
- `dit-prepare.h`'s `dit_prepare_encode_sample()` (step 2b): runs
  whenever `label.bpm <= 0 || label.keyscale.empty()`, decoupled from
  the caption-empty auto-label gate above it (Essentia doesn't need a
  caption to run).

**Verified**: matches confirmed ground truth (bpm=126, key=F minor)
exactly on a real track. `test-lora-roundtrip`/`test-lora-train-smoke`/
`test-model-store` still PASS.

## Training timestep distribution: continuous, not discrete

`src/dit-train.h` / `tools/ace-train.cpp`. Timestep sampling:
`t = sigmoid(N(mu=-0.4, sigma=1.0))`, drawn fresh every step, identical
for turbo/base/sft. Not the turbo inference schedule's 8 discrete
shift=3.0 values (`DIT_TRAIN_TURBO_SHIFT3_TIMESTEPS`, kept only for
`tests/test-lora-train-smoke.cpp`'s fixed-t overfit mechanism check).
"shift" is inference-schedule-only, never applied during training.
Source: `acestep-repo`'s `train.py` ships this as its `fixed` mode,
documented as reimplementing `sample_t_r()` from the model's own
`modeling_acestep_v15_turbo.py`.

CFG dropout: `--cfg-ratio` (default 0.15). With that probability, a
step's real conditioning is replaced by the model's own
`null_condition_emb` (already loaded for inference-time CFG, tiled
across every encoder position for training). Without it the LoRA never
sees the unconditional branch real CFG-guided generation evaluates.

Verified: `test-lora-train-smoke`/`test-lora-roundtrip` PASS unchanged.

## LoRA rank/alpha: 64/128, not 8/16

`acestep-repo`'s `train.py fixed` CLI defaults to `--rank 64 --alpha
128` (current recommended values in `training_v2`; rank=8/alpha=16
matches older docs/ComfyUI-FL and is superseded). At rank=8/alpha=16
with the corrected timestep+CFG-dropout recipe: no audible resemblance
at adapter_scale=1.0. At rank=64/alpha=128, same recipe: audible
resemblance at scale=1.0. Rank was necessary in addition to the
timestep/CFG fix -- rank=8 lacks capacity for the harder, more diverse
continuous-timestep+CFG-dropout objective (which itself converges to a
higher loss floor than the old discrete-schedule regime; this reflects
task difficulty, not regression).

Overtraining at rank=64 is fast and must be checked by ear, not by loss
alone: single ~75s sample, 3000-step cosine schedule (save every
250-500 steps to compare cheaply without retraining). One sample's
sweep: EMA~0.21 (good) -> EMA~0.10 ("drifting") -> EMA~0.06 ("obsessive"
/low quality). Sweet spot is roughly EMA 0.15-0.25 for this
duration/rank -- content-dependent, not a fixed step count.

Step count does not transfer across content directly: setting
`--epochs N` recalibrates the entire cosine LR schedule to decay by
step N. Comparing checkpoints from runs with *different* `--epochs`
budgets at the same step number is invalid (an early-decay run can look
"converged" at a step where its LR has already floored, while a
longer-budget run is still learning at that same step). Always compare
checkpoints against loss level and position in a consistently-scheduled
cosine curve, and use the same `--epochs` budget across content unless
you've confirmed a shorter one doesn't cut the LR schedule short.

## Generation flow: always `ace-lm` -> `ace-synth`

The standard, documented generation path (`docs/ARCHITECTURE.md`,
`README.md`) is always two-step: `ace-lm` generates real audio codes
from the caption, `ace-synth` renders them -- including for adapters
(README's worked example: a named external ComfyUI LoRA through this
exact flow). Skipping `ace-lm` and hand-setting `audio_codes: ""`
(pure text2music/silence context) is a different, non-standard path and
measurably lower quality; do not use it for LoRA A/B testing.

Training's `context_latents` are always silence+all-ones-mask
(text2music convention, `dit-prepare.h`) regardless of what generation
uses. This train/generation context difference exists but is not fatal
to LoRA quality in practice -- externally-trained community LoRAs work
fine through the standard `ace-lm` -> `ace-synth` flow despite it.

## GGML `ggml_mul_mat_set_prec(GGML_PREC_F32)`: no effect on this model

Investigated as a fix for small forward-pass drift vs PyTorch (temb's
`linear_1` output: cos=0.99998 vs Python). Verified via runtime dispatch
instrumentation in `ggml-cuda.cu` and source inspection, not guessing:

- Quantized weights (Q8_0, the actual production `synth_model`) dispatch
  to `ggml_cuda_mul_mat_q`/`ggml_cuda_mul_mat_vec_q`. Grepped
  `mmq.cu`/`mmq.cuh`/`mmvq.cu`/`mmvq.cuh` for `prec`/`GGML_PREC`/
  `op_params`: zero matches. These kernels never read the precision
  hint -- it is a no-op for every quantized matmul in the model.
- F32-weighted layers (temb) already default to F32 compute
  (`compute_type = src0->type`, and `src0` is F32 there), so forcing
  F32 changes nothing there either.
- The remaining ~cos 0.99-0.9998 per-layer drift vs Python is ordinary
  floating-point cross-implementation noise (different reduction
  order), not a bug -- confirmed by it being equally or more present in
  the no-adapter base case, which sounds correct.

`dit_ggml_linear`/`dit_ggml_linear_bias` do not set this precision hint.
Do not re-add it without first confirming (via dispatch logging, not
assumption) which CUDA kernel a given matmul shape/dtype actually hits.

## Gradient clipping: required at real sequence lengths

`dit_train_optimizer_step` (`src/dit-train.h`) now clips the *global*
L2 gradient norm across every LoRA tensor jointly, before the AdamW
step -- matches `torch.nn.utils.clip_grad_norm_`, which the Python
trainer applies via `--max-grad-norm` (default 1.0). New config field
`DiTTrainConfig::max_grad_norm` (default 1.0, matches Python; 0
disables), CLI flag `--max-grad-norm` in `ace-train fit`.

This was previously completely absent -- no norm computation, no
clipping, anywhere in the C++ trainer.

**Symptom this fixes**: real training runs (single ~75s sample,
matched hyperparameters/seed against the Python reference) showed EMA
loss flat/noisy around 0.85-1.05 for 1000 steps with no downward trend,
while Python's reference run on the identical sample/config/seed
dropped 0.84 -> 0.10 over the same 1000 steps.

**Isolated reproduction** (`test-lora-train-smoke --T <N> --enc-s <N>`,
fixed single (t, noise) target, no data/CFG/timestep-sampling variables
at all): overfit to near-zero loss at T=32/128/384 (small, synthetic
scale). At T=1872 (real 75s-sample scale) loss plateaus/oscillates
around 0.8-1.2 and never converges, even given far more steps than the
small-T cases needed. Confirms the instability is scale-dependent (T,
i.e. more attention positions contributing to the summed MSE loss and
its gradient), not data- or sampling-related, and reproduces outside
any real dataset. With clipping added, the same T=1872 fixed-target
test trends steadily downward instead of plateauing.

Diagnostic technique worth reusing: `test-lora-train-smoke`'s
fixed-target overfit isolates optimizer/gradient-flow correctness from
data and random-sampling variables, and its `--T`/`--enc-s` flags let
you reproduce scale-dependent issues without a real dataset.

## Reference audio for A/B comparisons lives in `/tmp/music`

Real training/reference tracks (e.g. `sasac_-_*`, `david_rubato_-_circuit.mp3`)
are in `/tmp/music`, not anywhere under `acestep-repo` or this repo's
`tmp/`. Files that *look* like preserved generation outputs elsewhere
(e.g. `acestep-repo/pytest_workspace/generated/*.mp3`) are not
verified ground truth just because their filename is plausible --
confirm what produced them before treating them as a reference.

## Python's `generate_music()`: `audio_codes` forces "cover" task

Passing `audio_codes` (used all session for deterministic forward-pass
numerical comparisons, since it removes the LM/codec sampling variable)
silently reassigns `task_type` to `"cover"` internally regardless of
the requested `task_type` -- heavily conditions generation on
reproducing that fixed reference's own structure. This is not the same
generation mode as judging whether a LoRA's *style* comes through.
For A/B-listening whether a LoRA learned anything, use plain
`text2music` with no `audio_codes` (generate from scratch, guided only
by caption/lyrics/metadata) -- that is what "does this LoRA sound like
the reference" claims are actually based on.

## Timestep sampling: `t = max` of two draws, not one

Python's `sample_timesteps()` (`training_v2/timestep_sampling.py`) draws
TWO independent `sigmoid(N(mu=-0.4, sigma=1.0))` samples per step and
assigns `t=max, r=min` -- then forces `r=t` anyway via
`data_proportion=1.0` (`use_meanflow=False` for every ACE-Step variant
during training). Net effect: `r` is discarded, but `t` itself is the
max of two draws, which is a different distribution than one draw
(skews higher on average). `tools/ace-train.cpp`'s `sample_t` now
matches this exactly (draws two, takes max) instead of a single draw.

## LoRA-A init: PEFT's `kaiming_uniform_(a=sqrt(5))`, not `N(0, 1/sqrt(fan_in))`

Verified via PEFT source (`peft/tuners/lora/layer.py`,
`reset_lora_parameters`): default init is
`nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))`. For a Linear
weight this reduces algebraically to `Uniform(-1/sqrt(fan_in),
1/sqrt(fan_in))` (gain=sqrt(2/(1+5))=sqrt(1/3), bound=sqrt(3)*gain/sqrt(fan_in),
and sqrt(3)*sqrt(1/3)=1 cancels). The Gaussian `N(0, 1/sqrt(fan_in))`
used previously has ~1.73x (sqrt(3)x) the standard deviation of PEFT's
actual init. `dit_train_alloc_proj` (`src/dit-train.h`) now draws
Uniform to match exactly. B stays zero-init either way (both agree).

## Gradient clipping + timestep fix + init fix: measured effect

Same single ~75s sample, matched hyperparameters/seed as Python,
comparing EMA loss at each of Python's own logged checkpoints:

| step | Python | ours: no fixes | ours: +clip | ours: +clip+timestep+init |
|------|--------|-----------------|--------------|------------------------------|
| 250  | 0.844  | ~0.95 (flat)     | 0.939        | 0.939 |
| 500  | 0.537  | ~0.95 (flat)     | 0.777        | 0.705 |
| 750  | 0.369  | ~0.83 (flat)     | 0.592        | 0.616 |
| 1000 | 0.099  | 0.888            | 0.613        | 0.556 |

Gradient clipping is the dominant fix (flat -> steadily decreasing).
Timestep/init fixes are real (verified via source, see above) but
contribute less at this step count. A real gap to Python's step-1000
value remains at this schedule length -- see next section for why, and
why it is not a blocker.

## Cross-language single-step gradient comparison: real, characterized, not step-count-fixable

`tests/test-grad-compare.cpp` dumps a single training step's LoRA
gradient (real sample, fixed LoRA A/B init + noise/t, exported to raw
files) for cross-checking against Python's actual
`FixedLoRAModule.training_step()` computing the identical step
(`tmp/grad-compare/compare.py`). Findings, all via direct measurement:

- Forward pass matches closely everywhere tested: loss values agree to
  ~5 decimal places, at every layer depth and every data configuration
  below.
- Small-magnitude data (any T, any enc_S, including the real sample's
  exact T=1876/enc_S=109 dimensions with *synthetic* small values):
  final-layer LoRA gradient cos_sim = 0.96-0.999.
- Real data (actual VAE-encoded latents + actual conditioning, values
  up to +-59): final-layer (layer0, closest to the LoRA) gradient
  cos_sim = 0.04-0.05 -- effectively uncorrelated.
- Per-layer trace on the full 24-layer model (`--dump-layer-grads`,
  `tmp/grad-compare-layers/compare_layers.py`, gradient hooks +
  `retain_grad()` on every layer): cos_sim is 0.9995 at layer23 (last
  layer, closest to the loss) and degrades smoothly backward through
  the stack to 0.047 at layer0 -- a compounding accumulation through
  24 layers of backward propagation, not a single localized bug at one
  layer. (An earlier claim of a hard break at "layer index 3" was a
  methodology artifact of capping the model to N layers, which changes
  the architecture each time since the real output needs all 24 layers
  plus the final AdaLN+proj_out stage -- retracted.)
- Ruled out via direct experiment, not reasoning: PyTorch `eager` vs
  `sdpa` attention implementation for the comparison (identical result,
  rules out "different attention algorithm"); a buffer-reuse bug in the
  *diagnostic tool itself* (found and fixed -- `ggml_set_output` must be
  called on the gradient tensor, not just the forward tensor, or the
  allocator recycles its buffer before it's read back); GGML's softmax
  backward reduction accumulating in `float` instead of `double`
  (`ggml-cuda/softmax.cu`, `ggml-cpu/ops.cpp` -- patched to match
  `ggml_rms_norm_back`'s existing double-precision convention as a
  correctness improvement regardless, but it measured zero effect on
  this specific divergence: 0.047071 vs 0.047192, noise-level).
- Finite-difference verification (perturb a LoRA weight, compare
  numerical vs analytic gradient) is unreliable at this gradient
  magnitude (~1e-6): tested against a case with *known-good* cos_sim
  (0.96+ at small-T) and it disagreed there just as badly as at real
  scale, proving the check itself -- not the gradient -- is the
  unreliable part at this magnitude in F32. Retracted as inconclusive.

**Practical resolution**: this is real, measured, gradient-direction
imprecision that compounds over the 24-layer stack when real
(wide-dynamic-range) data is used, not (as far as every specific op's
backward formula checked by direct source reading) a single fixable
formula bug. AdamW is tolerant of gradient noise -- it just needs more
steps to average it out. Confirmed empirically: the same single sample,
same 3 fixes, run for 2500 epochs instead of 1000 (recalibrating the
whole cosine schedule, per the step-count-doesn't-transfer note above)
reaches EMA loss 0.115-0.160 by steps 2000-2500, in Python's own
convergence range (Python's longer runs reached 0.069-0.075 around
epochs 1500-1750). More steps, not a code fix, closes the remaining gap
in practice.

## Cross-attention "root cause" retracted: it was a test-harness bug

An earlier pass of this investigation used isolated backward tests
(seed Python's autograd with our own computed upstream gradient at a
specific tensor, then check whether Python's backward reproduces our
downstream gradient -- isolates one op/sub-step from everything else)
and concluded the divergence was localized to cross-attention's core
`Q@K^T -> softmax -> probs@V` computation, specifically via
`ggml_out_prod` (the op `ggml_mul_mat`'s backward routes through) and
its GQA-broadcast handling. Two double-precision patches were made
against this theory (see below) and measured zero effect -- which,
in hindsight, should have been the signal to question the localization
itself rather than the precision hypothesis.

**The actual bug**: the manual test scripts that reimplemented
`AceStepDiTLayer.forward()`'s cross-attention stage by hand
(`isolate_subcomponents.py`, `isolate_cross_attn.py`) fed the RAW
condition-encoder output directly into `layer.cross_attn(...)`. Python's
real model applies `self.condition_embedder` (a `Linear`, matching our
own `cond_emb_w`/`cond_emb_b`) to this tensor exactly ONCE, at the top
of `AceStepDiTModel.forward()`, before ever reaching a layer
(`modeling_acestep_v15_turbo.py:1363`). Scripts that call individual
layers/sub-modules directly (bypassing the model's own `forward()`)
must apply this projection themselves; scripts that hook the model's
real forward pass (`compare_layers.py`, `isolate_layer1.py`) get it for
free and were never affected.

**Verified fix and correction** (`tmp/grad-compare-fwd/check_forward_qkv.py`):
applying `decoder.condition_embedder` correctly makes cross-attention's
forward AND backward match near-perfectly:
- `k_proj_out`/`v_proj_out` forward values: cos_sim 1.000000 (were
  falsely 0.00/-0.06 -- a scale error of ~6x, from feeding
  un-projected data into a projection-sized weight matrix).
- Sub-component backward, corrected (`isolate_subcomponents.py`
  re-run): cross-attention output cos_sim=0.999989 (was falsely
  0.757); the real, modest divergence is in **self-attention's**
  backward instead (0.9999 -> 0.906 crossing it), matching the
  ~0.90-0.96 "steady-state" level seen throughout the 24-layer trace,
  not a localized catastrophic failure anywhere.

**Corrected conclusion**: there is no single identifiable formula bug.
Every backward formula actually checked by direct source reading
(softmax, RMSNorm, MUL_MAT/out_prod, the view/reshape/permute/transpose
family) is mathematically correct. The double-precision changes to
softmax's backward reduction (`ggml-cuda/softmax.cu`,
`ggml-cpu/ops.cpp`) and the experimental double-precision `out_prod`
path (`ggml-cuda/out-prod.cu`, gated behind `GGML_OUT_PROD_F64_ACCUM=1`,
off by default) are kept as genuine correctness improvements matching
`ggml_rms_norm_back`'s existing convention, but neither was "the fix"
-- both were correctly measured as having no effect, because the
localization that motivated them was itself wrong. The real
divergence is a modest (~5-10%), not-yet-further-localized per-layer
gradient-direction difference that compounds smoothly over the
24-layer stack specifically when real (wide-dynamic-range) data is
used -- ordinary cross-implementation floating-point behavior at this
depth, not a bug found so far. AdamW tolerates it; more optimizer
steps closes the gap in practice (see above, confirmed to Python's own
convergence range at 2000-2500 steps).

**Methodological lesson, worth keeping**: any test script that
reimplements part of a model's forward pass by hand (rather than
hooking the model's own `forward()`) must be checked line-by-line
against the *entire* real forward path, including steps that happen
once outside the immediate module being tested -- not just the
module's own documented inputs/outputs. A script that "looks like" it
reproduces one layer's computation can silently skip a shared,
upstream projection and produce a confident, precisely-reproducible,
false localization.

## Python trains the LoRA in bf16, not f32 -- confirmed via source, not yet portable

Read directly from `training_v2/trainer_fixed.py` (`FixedLoRATrainer`,
the class `train.py fixed` actually instantiates -- confirmed via
`cli/train_fixed.py`'s import, this is the real production path, not a
dead code branch):

- `self.module.model = self.module.model.to(self.module.dtype)`
  (line 377) casts the **entire decoder, including LoRA A/B**, to
  `self.module.dtype` -- `bf16` on CUDA (`_select_compute_dtype`).
- Wrapped in `self.fabric.setup(...)` with Lightning Fabric's
  `"bf16-mixed"` precision plugin.
- Critically, `acestep/training/trainer.py` (the *older*, unused
  trainer) has an explicit safety net missing here: it checks
  `if device_type == "mps" or precision.endswith("-mixed")` and, in
  that case, casts to **fp32** instead, then calls
  `_ensure_trainable_params_fp32()` to force any bf16-drifted trainable
  tensor back to fp32 before the optimizer touches it. `trainer_fixed.py`
  has no such check and no equivalent call anywhere in `training_v2/`.
- `fixed_lora_module.py`'s `training_step()` additionally wraps the
  forward pass in `torch.autocast(dtype=bf16)`. PyTorch's autocast
  policy (documented, not project-specific) casts matmul-family ops
  (`linear`, `matmul`, `conv*`) to bf16 while automatically keeping
  softmax/layer_norm/reductions in fp32 -- so the real per-op precision
  split is "matmuls in bf16, norms in f32", not "everything bf16" or
  "everything f32".

Our C++ trainer runs matmuls, norms, and gradients entirely in F32.
This is a real, source-confirmed precision mismatch against the actual
production trainer (not the F32-forced Python used for every gradient
comparison above, which was never representative of what `train.py
fixed` actually runs) and is a more plausible dominant explanation for
the residual per-layer gradient-direction gap than anything in the
cross-attention/out_prod investigation.

**Attempted fix, reverted -- hard architectural blockers found by
building it, not by guessing**:

1. Tried fake-quantizing activations/weights through a
   `ggml_cast(F32->BF16->F32)` round-trip before every
   `dit_ggml_linear`/`dit_ggml_linear_bias` matmul (gated to training
   only via a flag, zero effect on inference). Assumed `ggml_cast`
   needed a new backward case; it doesn't -- `ggml_cast` actually
   builds a `GGML_OP_CPY` node (confirmed via `ggml_cast`'s source,
   `ggml.c:3547`), and `GGML_OP_CPY` already has a working
   straight-through-style backward. The wrong-opcode addition didn't
   even compile and was removed.
2. With that removed, the round-trip still **crashes**:
   `ggml_build_backward_expand` asserts
   `node->src[j]->type == GGML_TYPE_F32 || GGML_TYPE_F16` for every
   node in the built backward graph (`ggml.c:7273`) -- BF16 is not an
   accepted gradient dtype anywhere in this fork's autodiff, full stop.
   Confirmed by running it, not by reading alone.
3. The correct fix needs a value transform that rounds F32 data to
   BF16 *precision* while staying F32-*typed* (so it never trips the
   above assertion) -- i.e. a new elementwise op, since this isn't
   expressible as arithmetic composition of existing ops (BF16
   truncation is a bit-level operation, not a smooth function) and
   `ggml_map_custom1` (GGML's generic escape hatch for arbitrary
   elementwise C code) has neither a CUDA implementation nor a backward
   case registered anywhere in this fork.
4. `ggml_out_prod` (`MUL_MAT`'s backward) and `ggml_opt_step_adamw`
   both hard-assert F32 for every operand on both CPU and CUDA --
   confirmed via source in both `ggml-cpu` and `ggml-cuda`. Neither
   accepts BF16 today regardless of the above.

**Status**: reverted rather than ship an unvalidated new core op
(bit-level BF16 round-trip, CPU+CUDA kernels, backward wiring) touching
shared dispatch tables used by every other op in the codebase, in the
same session as everything else already changed. The concrete next
step for genuine bf16-matmul parity is that new op -- forward: truncate
an F32 value's mantissa to BF16 precision, stay F32-typed; backward:
straight-through (gradient passes unchanged) -- plus wiring it into
`dit_ggml_linear`/`dit_ggml_linear_bias` behind the training-only flag
already scoped out above. This is additive (new op, existing behavior
untouched) and should not require touching `ggml_out_prod` or
`ggml_opt_step_adamw` at all, since the round-tripped tensor stays F32
the whole time and both of those already accept F32.
completely wrong localization.

## Listening-test checkpoint: cpp vs python LoRA, same trigger/dataset

Both trained on the same single-sample dataset (`acetrig01`, rank=64,
alpha=128), both generated through each side's own real, unmodified
pipeline (`ace-lm` -> `ace-synth` for cpp; `generate_music()` +
`add_lora()`/PEFT for python) -- same caption, bpm, keyscale,
timesignature, no shared/fixed audio codes, no cover-mode.

User's verdict: outputs are similar; python's is a little higher
quality. Consistent with the still-open bf16-matmul-training gap
documented above (python trains matmuls in bf16 under autocast, cpp
trains everything in F32) -- not yet re-diagnosed further, just noted
as the leading candidate for the remaining gap.

## `ggml_bf16_rtne`: bf16-precision matmul op, implemented

Built the op scoped above instead of the full-tensor-dtype approach that
crashed. New `GGML_UNARY_OP_BF16_RTNE` (`ggml.h`/`ggml.c`, CPU
`unary-ops.cpp`+`ops.cpp`+`ggml-cpu.c`, CUDA `unary.cu`+`unary.cuh`+
`ggml-cuda.cu`):

- Forward: F32 value -> round-to-nearest-even to BF16 precision -> back to
  F32 (same bit algorithm as `ggml_compute_fp32_to_bf16`, reimplemented
  `__device__`-side for CUDA via `__float_as_uint`/`__uint_as_float` since
  the host `static inline` version isn't callable from device code).
  Tensor stays F32-typed throughout, so it never touches
  `ggml_build_backward_expand`'s F32/F16-only assertion, `ggml_out_prod`,
  or `ggml_opt_step_adamw` -- all three still see plain F32.
- Backward: straight-through estimator, `ggml_add_or_set(isrc0, grad)`
  (rounding's true derivative is zero a.e.; STE passes it through
  unchanged, standard QAT practice).

Wired into `dit_lora_delta` (`src/dit-graph.h`) only -- the LoRA A/B
branch's own matmuls (`x`, `A`, hidden `h`, `B` all rounded immediately
before each `dit_ggml_linear` call), gated by a global
`g_dit_train_bf16_lora` flag set true only around the graph-build calls
in `dit_train_forward_backward`/`dit_train_eval`, false everywhere else
(inference, adapter merge). Deliberately does NOT touch the frozen base
weights' matmuls: those already load in their native GGUF dtype (BF16 for
the main projection weights, confirmed earlier), so the base branch was
never the F32 outlier -- only the trainable LoRA branch was.

Retrained (rank=64, alpha=128, 1500 epochs, same single-sample dataset):
EMA loss 0.377, matching the F32 run's 0.369 -- no NaN, no crash, no
meaningful loss-curve regression from adding the rounding. Generated
another cpp-vs-python comparison through each side's real pipeline;
listening verdict pending.

## Second-material test: `shook_official_-_papaya.mp3` (nu-disco, trigger `papayatrig01`)

Repeated the full cpp-vs-python trained-LoRA comparison on a second,
unrelated reference track (`/tmp/music`) to confirm the circuit-track
result wasn't a fluke of that one sample. 75s excerpt (30s-105s),
metadata (caption/bpm=122/keyscale=A minor/timesig=4) derived via our
own `ace-understand` tool once and reused identically for both
trainers' dataset sidecars, rank=64/alpha=128, 1500 epochs both sides.

Two real environment/tooling issues hit and fixed along the way,
neither specific to this repo's code:

1. `acestep-repo/acestep/training/dataset_builder_modules/preprocess_audio.py`
   called `torchaudio.load()` directly with no fallback; this
   environment's `torchcodec` can't find any working ffmpeg
   (`libavutil.so.56/57/59/60` all missing) and `torchaudio.load` no
   longer has its own legacy backend, so every preprocess call raised
   `RuntimeError: Could not load libtorchcodec`. Patched in a
   soundfile-based fallback (mirroring the fallback already present
   next door in `dataset_builder_modules/audio_io.py::get_audio_duration`,
   so this is completing an existing pattern, not inventing one).
2. `train.py fixed --preprocess ...` only preprocesses and exits (its
   final log line literally says "You can now train with: ...") --
   it does not chain into training even though `--dataset-dir`/
   `--output-dir`/training hyperparams are accepted on the same
   invocation. Preprocessing and training are two separate CLI calls.
   Also: `train.py fixed` (without `--preprocess`) prints a parameter
   summary and blocks on an interactive `Start training? [Y/n]`
   prompt with no non-interactive flag -- needs `yes |` piped into
   stdin for scripted/background runs.

Both trainers converged cleanly (cpp EMA loss 0.139, python final-epoch
loss 0.082), no NaN/crash on either side, on a track with a
substantially different genre/BPM/key than the first test.

## Why python's reported training loss number looks lower: not a bug, probably RNG variance

User noticed python's final loss (0.082) is clearly lower than cpp's
(0.139 EMA) at the same step count and asked why. Compared the two
loss computations line-by-line before concluding anything:

- `fixed_lora_module.py::training_step` (python):
  `x1=randn_like(target)` (noise), `x0=target_latents` (data),
  `xt = t*x1 + (1-t)*x0`, `flow = x1 - x0`,
  `loss = F.mse_loss(v_pred, flow)` (mean over all elements).
- `dit-train.h` (`dit_train_forward_backward`/`dit_train_eval`, cpp):
  identical convention (`xt = t*x1+(1-t)*x0`, `target=x1-x0`,
  `loss = sum((v_pred-target)^2) / nelements(v_pred)`, i.e. the same
  mean reduction). Formulas match exactly; this is not a
  reduction/scaling bug.

But the two numbers being compared are NOT computed from the same
data: every step draws its own fresh random noise `x1` and timestep
`t` (`sample_timesteps`: `max(sigmoid(N(-0.4,1)), sigmoid(N(-0.4,1)))`),
and cpp's RNG (`std::mt19937`) and python's (`torch`'s generator) are
different algorithms -- same seed=42 does NOT produce the same
sequence across them. So "loss at step 1500" (or its EMA) from each
trainer is one point on two *different, independently-random*
trajectories, not two measurements of the same thing. The
flow-matching objective's per-step loss variance is itself
timestep-dependent (steps landing near `t=1`, i.e. `xt`~pure noise,
structurally carry different squared-error scale than steps near
`t=0`, regardless of model quality) -- both logs show this directly:
cpp's own per-step loss swings from ~0.08 to ~0.9 step to step late in
training, on a single unchanging model. A single final-step (or even
a 10-step-EMA) comparison between two independent random walks is
mostly comparing noise to noise, not adapter quality.

Consistent with this: immediately after seeing the loss numbers, a
second listening round on the SAME two adapters had the user's
verdict flip the other way (cpp sounded better than python on the
first round's files). Loss curves from single-sample, highly
stochastic per-step training are not a reliable proxy for perceptual
adapter quality here -- listening is. Not chasing this further as a
"cpp is worse" bug without more direct evidence (e.g. matching RNG
streams for an apples-to-apples curve) than one final number.

## Consolidation pass: fixed a real production-path leak, wrote the user tutorial

Before writing a user-facing tutorial, audited every diagnostic addition
from this development arc for whether it could affect ordinary generation
(`ace-lm`/`ace-synth`/`ace-server`), not just training. Found one real leak:

- `src/dit-graph.h`'s per-layer/per-sub-component debug dumps (`dbg_out`,
  `dbg_ca_out`, and the `hidden_after_layer{N}` block added earlier this
  arc for `tests/test-grad-compare.cpp`'s backward-pass localization) were
  **unconditional** inside `dit_ggml_build_graph` -- the same function
  `src/dit-sampler.h` calls for every real generation step. Every
  `ace-synth` call was building dozens of extra named+`ggml_set_output`
  tensors for layers 0-3 (plus one per layer for `hidden_after_layerN`)
  that only the diagnostic tool ever reads. `ggml_set_output` pins a
  tensor's buffer for the graph's lifetime, so this was a real (if
  probably small at these tensor sizes) memory/allocator overhead added to
  production inference by test-only code, not just visual clutter.
- Fixed: added `g_dit_debug_layer_dumps` (default `false`), gating all
  three sites; `tests/test-grad-compare.cpp` sets it `true` only around its
  own `dit_ggml_build_graph` call. Rebuilt everything, confirmed a plain
  `ace-lm`->`ace-synth` generation still runs clean with no behavior
  change, and confirmed `test-grad-compare --dump-layer-grads` still finds
  every tensor it needs with the flag on.

Remaining training-only surface (`g_dit_train_bf16_lora`, the
`ggml_bf16_rtne` op itself) was already correctly gated -- verified by
grepping every call site, not assumed.

Wrote [`docs/TRAINING.md`](docs/TRAINING.md): a practical, user-facing
tutorial (prepare -> fit -> generate, recommended rank/alpha/epochs,
trigger-word convention, the `audio_codes`-forces-cover gotcha, and a
troubleshooting section) -- deliberately separate from this file, which
stays a development log. Linked from `README.md`'s Adapters and Technical
documentation sections.

## Full split: `src/dit-graph.h` reverted to pristine, `src/dit-train-graph.h` added

Follow-up request: don't patch acestep.cpp's own files for training at
all, even the gated version above -- prefer duplication, since a shared
file like `dit-graph.h` is exactly the kind of change an upstream project
would be reluctant to take. Confirmed via git that every piece of
LoRA/training code in `dit-graph.h` (the `DiTLoraProj`/`DiTLoraLayer`
structs, `dit_lora_delta`, `dit_ggml_add_lora`, the `lora`/`lora_layers`
parameters threaded through every builder, `want_grads`, and this
session's `g_dit_train_bf16_lora`/`g_dit_debug_layer_dumps` flags) was
100% added by this training branch -- `master`'s `dit-graph.h` has none of
it. Also confirmed `src/dit-sampler.h` (the real generation path) calls
`dit_ggml_build_graph` with **no** `lora_layers` argument at all --
production adapters are applied by merging into GGUF weights at load time
(`src/adapter-merge.h`), never through this graph-level mechanism. So none
of it was load-bearing for generation in the first place.

Action: moved the entire LoRA/bf16/debug-dump graph-building layer into a
new, fully self-contained `src/dit-train-graph.h` (duplicates the
non-LoRA architecture code too -- temb, self-attn, cross-attn, MLP,
per-layer composition -- rather than including `dit-graph.h`), then
`git checkout master -- src/dit-graph.h` to restore it byte-for-byte.
Verified: `git diff master -- src/dit-graph.h` is empty; `grep` for
`DiTLoraLayer`/`g_dit_train_bf16_lora`/`g_dit_debug_layer_dumps` outside
`dit-train-graph.h`/`dit-train.h`/`test-grad-compare.cpp` finds nothing.
Rebuilt everything; `ace-lm`->`ace-synth` smoke test, `test-lora-train-smoke`,
and `test-grad-compare --dump-layer-grads` all still pass.

`docs/TRAINING.md`'s design-note section rewritten to describe the new
split accurately (was written for the gated-flag version, now stale within
the same session).

Known remaining coupling, flagged but not touched without direction:
`tools/ace-understand.cpp` (general-purpose CLI, not training-exclusive)
now also uses the new Essentia-based BPM/key correction
(`src/audio-analysis.h`) -- a real improvement to that tool's own output,
not training-specific machinery, so left as-is pending explicit
direction. `CMakeLists.txt`/`tests/CMakeLists.txt` additions (registering
the `ace-train` and `test-grad-compare` targets) are unavoidable build-system
wiring for a new binary, not behavioral changes to existing code.

## Final resolution: `ggml_bf16_rtne` as a build-time patch, not a submodule commit

Follow-up constraint, stricter than the one above: no commits into the
`ggml` submodule's own git history at all -- not even isolated, documented
ones. This ruled out the original `ggml_bf16_rtne` implementation as
originally shipped (a real commit on the submodule's local `master`).

Two submodule-free alternatives were built, measured, and rejected before
landing on the final approach -- kept here for the record so they aren't
reinvented:

1. **`ggml_map_custom1` + `ggml_backend_sched`.** `ggml_map_custom1` is
   GGML's own public escape hatch for arbitrary host callbacks, so the
   rounding forward pass needs zero submodule changes. But it has no CUDA
   kernel and no backward case anywhere in this fork, so it only works via
   `ggml_backend_sched`'s automatic CPU fallback (confirmed already used
   in `src/dit-sampler.h`, so not a foreign pattern) -- correct
   (`test-lora-train-smoke` loss trajectory matched to 5-6 significant
   digits), but every rounding op running off-GPU cost ~2x per training
   step (measured: 1500 epochs went from ~10-13 min to projected 20-30+
   min).
2. **Two chained `ggml_cast` calls (F32->BF16->F32) + a transient `type`
   spoof.** `ggml_cast` is a genuine, unmodified, native CUDA op already
   used throughout this codebase for weight dtype conversion, so this is
   fully native -- no CPU, no scheduler. The blocker:
   `ggml_build_backward_expand` hard-asserts any gradient-needing source
   is F32/F16 (`ggml.c`: `GGML_ASSERT(node->src[j]->type == GGML_TYPE_F32
   || GGML_TYPE_F16)`), and the intermediate BF16 tensor trips it. Fix:
   temporarily relabel that one intermediate tensor's `type` field to F32
   for the sole duration of the `ggml_build_backward_expand()` call, then
   restore it -- verified safe by reading `ggml_reshape`'s source (the one
   thing CPY's backward formula calls on it): it only asserts on
   `ggml_nelements`, and its own comment states "only the shape of b is
   relevant, not its memory layout." This produced loss numbers matching
   to 5-6 significant digits and needed zero CPU/scheduler, but full
   per-matmul fidelity (rounding right before every individual q/k/v/o
   LoRA matmul, matching PyTorch autocast's literal per-op behavior) needs
   ~625 of these round-trips per training step -- ~1250 extra graph nodes
   in a graph that's rebuilt from scratch every epoch, which inflates
   `ggml_build_backward_expand`'s and the allocator's per-step,
   node-count-scaling cost. A pointer-identity dedup cache (several call
   sites round the exact same tensor -- self-attn's q/k/v share one
   `norm_sa`, all 24 layers' cross-attn k/v share one graph-wide `enc`)
   cut this from ~0.74s/epoch to ~0.57s/epoch (measured, 500-epoch runs),
   still ~15-40% slower than the ~0.4-0.5s/epoch original baseline, purely
   from graph size, not CPU/scheduler involvement this time.

Both were rejected as compromises, not because they were broken. Given
the explicit "we want the original method back, positioned in our code
instead of patched into ggml" requirement, the actual final answer was
simpler than either: **keep the original op exactly as built, store it as
a patch file in this repo instead of a submodule commit.**

`patches/ggml-bf16-rtne.patch` (see `patches/README.md`) contains the
identical diff as the original working implementation --
`GGML_UNARY_OP_BF16_RTNE`, native CUDA + CPU kernels, real backward case.
`CMakeLists.txt` applies it to the `ggml` submodule's *working tree* at
configure time, before `add_subdirectory(ggml)`, guarded by
`git apply --reverse --check` for idempotency. The submodule's own commit
history is never touched -- verified by reverting the submodule to
pristine (`git checkout -- .` inside `ggml/`) and confirming `cmake`
reapplies the patch cleanly from scratch, twice (first apply, then a
no-op reconfigure that correctly detects "already applied").

Verified after restoring this: `test-lora-train-smoke`'s loss trajectory
is byte-for-byte identical to the original implementation's (every digit
shown matches -- step 0: 2.824832, step 9: 1.085713, avg
1.918640->1.232062). Timed 500-epoch run: 4m22.113s total ->
~0.494s/epoch (subtracting ~15s model load), squarely inside the original
~0.4-0.5s/epoch baseline. Zero compromises: native speed, correct
backward, no submodule commit.

`src/dit-train-graph.h`/`src/dit-train.h` are back to the simplest form
(`ggml_bf16_rtne(ctx, t)` called directly, gated by
`g_dit_train_bf16_lora`) -- no double-cast, no type-spoofing, no
scheduler, no dedup cache; those all existed only during the
rejected-alternative phase and are gone from the current code.
