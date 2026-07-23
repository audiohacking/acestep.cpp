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

New binary: **`ace-train`** (tools/ace-train.cpp), two subcommands sharing
the models dir and ModelStore infra:

```
# 1) Dataset preparation: decorate + encode
./ace-train prepare \
    --models models \
    --dataset  path/to/audio_dir \        # .wav/.mp3 (+optional sidecars)
    --output   path/to/tensors_dir        # per-sample GGUF + manifest.json

# 2) Training
./ace-train fit \
    --models   models \
    --tensors  path/to/tensors_dir \
    --output   adapters/my-lora \
    --rank 8 --alpha 16 --lr 1e-4 --epochs 100 --grad-accum 4 \
    --save-every 10 --seed 42 [--val-split 0.1]
```

### prepare stage

Per audio file in `--dataset`:
1. Sidecar discovery (Python-compatible): `{name}.lyrics.txt`/`{name}.txt`
   (lyrics), `{name}.json` (caption/bpm/keyscale/timesignature/language),
   `{name}.caption.txt`.
2. Missing metadata → **understand pipeline** (`pipeline-understand`) fills
   caption, lyrics, bpm, keyscale, duration, language. This is the
   "Auto Label" step, native.
3. Encode (all existing modules): audio → 48kHz stereo → VAE encode →
   `target_latents`; build SFT prompt → BPE → Qwen3 encoder → text_hs;
   lyrics → BPE → embed_tokens → lyric_hs; CondEncoder(text, lyric,
   timbre=zeros) → `encoder_hidden_states` (+mask); `context_latents` =
   concat(silence_latent[:T], ones).
4. Write one GGUF per sample (`{name}.tensors.gguf`) holding the 5 tensors
   + metadata KV (caption, source path, duration); append to `manifest.json`.

GGUF as tensor cache = zero new formats, existing reader infra, easy to
inspect with existing tooling.

### fit stage

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
shapes. Grad clipping (Python's `max_grad_norm=1.0`) is not implemented yet;
not needed for the Phase 2 overfit check, revisit in Phase 3.

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
- [ ] **Phase 3 — `ace-train fit`**
  Full loop: dataset iteration, shuffling, grad accum, LR schedule,
  checkpoints, val split, logging.
- [ ] **Phase 4 — `ace-train prepare`**
  Sidecar scan + understand decoration + tensor GGUF cache. (After fit so
  early fit testing can use hand-built tensors from the existing pipelines.)
- [ ] **Phase 5 — end-to-end validation**
  Train a small-style LoRA on 10–20 songs; A/B the adapter in ace-synth /
  WebUI vs base model; compare loss curves against the Python trainer on the
  same dataset (both start from the same BF16 turbo DiT).
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
  gradient) along the way — see Decisions log. Next up: Phase 3, the
  `ace-train fit` CLI (dataset iteration, LR schedule, checkpoints).

## Verification checklist (running)

- [x] Saved safetensors loads via adapter-merge with correct alpha/rank scaling
      (`tests/test-lora-roundtrip.cpp`, 192/192 tensors, cossim 0.999982)
- [x] Backward graph builds without abort for full 24-layer DiT (CUDA;
      CPU backend not yet exercised, no CPU-only dev machine used so far)
- [x] LoRA A/B grads nonzero after 1 step; frozen weights bit-identical
      (`tests/test-lora-train-smoke.cpp`)
- [x] Overfit single sample: loss → ~0 (1.78 → 0.000047 over 150 steps,
      full 24-layer model)
- [ ] Same backward graph verified on CPU backend specifically
- [ ] soft_max_ext backward correct with a *non-trivial* attention mask
      (smoke test uses all-valid masks; real padding not yet exercised)
- [ ] VRAM measured at production T (~2000+ latent frames)
- [ ] prepare-stage tensors match Python preprocessed .pt (cossim) on one sample
- [ ] Trained LoRA audibly shifts style in ace-synth output
