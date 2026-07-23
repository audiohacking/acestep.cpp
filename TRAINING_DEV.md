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

- `ggml/include/ggml-opt.h`: full training module. AdamW + SGD, MSE loss,
  gradient accumulation (`opt_period`), per-step optimizer params callback
  (implements warmup+cosine LR host-side), dynamic-graph mode
  (`ggml_opt_prepare_alloc` per step) which handles variable song lengths
  without padding.
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

1. Load DiT from GGUF, dequant weights on the backward path to F32
   (constraint 1a). Frozen — never touched by the optimizer.
2. Create LoRA tensors A[r, in] (init: kaiming/normal·0.01) and B[out, r]
   (init: zeros) in F32 for each target projection in each layer;
   `ggml_set_param` on all of them.
3. Build training graph per sample length (dynamic-graph ggml-opt mode):
   host computes xt, flow target, samples t (std::mt19937 seeded by
   config.seed) → inputs; graph = DiT forward (no-FA, split-swiglu,
   LoRA branches) → outputs `v_pred`; labels = flow; loss = MSE.
4. Optimizer loop: `ggml_opt_alloc(backward=true)` + `ggml_opt_eval` per
   sample; `opt_period = grad_accum`; LR callback implements linear warmup
   → cosine annealing (eta_min = lr·0.01). Log per-step loss + EMA to
   stderr (SSE-friendly single-line format for later server integration).
5. Checkpoints every `--save-every` epochs + `final/`:
   `adapter_model.safetensors` (PEFT key naming
   `base_model.model.layers.N.{self_attn,cross_attn}.{q,k,v,o}_proj.lora_{A,B}.weight`)
   + `adapter_config.json` (r, lora_alpha, target_modules). Must load
   unmodified through `src/adapter-merge.h` → instantly usable in the WebUI.
6. Optional `--val-split`: held-out samples evaluated per epoch (forward
   only), best checkpoint tracked.

Grad clipping (max_grad_norm=1.0): ggml-opt has no built-in clipping; read
back LoRA grads (`ggml_opt_grad_acc`), compute global norm host-side, scale
LR for that step accordingly (equivalent effect), or add clip to fork later.
Total LoRA params at r=8 on 24 layers ≈ 24 layers × 8 proj × (2048·8+8·2048)
≈ 6.3M params → grad readback is cheap.

### Phase plan

- [x] **Phase 0 — study & feasibility** (this document)
- [ ] **Phase 1 — safetensors writer + LoRA checkpoint round-trip**
  Write adapter_model.safetensors from synthetic A/B tensors, reload via
  adapter-merge, verify merged delta numerically. Small, de-risks the output
  format first.
- [ ] **Phase 2 — training graph + backward smoke test**
  DiT forward with LoRA branches on 1 tiny sample; build backward via
  ggml-opt; confirm no unsupported-op aborts (CPU first, then CUDA); confirm
  loss decreases on an overfit-one-sample test. Measure VRAM vs T.
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

## Progress log

- 2026-07-23: Studied Python trainer + tutorial + preprocessing modules.
  Audited GGML fork autograd/opt coverage against the DiT graph op
  inventory — **no fork changes required to start**. Branch `training`
  created, plan written.

## Verification checklist (running)

- [ ] Backward graph builds without abort for full 24-layer DiT (CPU)
- [ ] Same on CUDA; scheduler places OUT_PROD/OPT_STEP_ADAMW correctly
- [ ] soft_max_ext backward correct with attention mask input
- [ ] LoRA A/B grads nonzero after 1 step; frozen weights bit-identical
- [ ] Overfit single sample: loss → ~0
- [ ] Saved safetensors loads via adapter-merge with correct alpha/rank scaling
- [ ] prepare-stage tensors match Python preprocessed .pt (cossim) on one sample
- [ ] Trained LoRA audibly shifts style in ace-synth output
