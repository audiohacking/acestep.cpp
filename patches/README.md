# patches/

Local patches applied to the `ggml` submodule's working tree at CMake
configure time — see the "Local ggml patches" block near the top of
`CMakeLists.txt` (right before `add_subdirectory(ggml)`).

**These are never committed into the `ggml` submodule's own git history.**
The submodule stays at its normal upstream commit; `git apply` just
modifies the checked-out working tree files before they're compiled, the
same way any local, uncommitted edit would. Re-running `cmake` is
idempotent — it checks (`git apply --reverse --check`) whether a patch is
already applied and skips it if so, so repeated configures, incremental
builds, and CI don't fail on a "patch already applied" error.

## Why patches instead of a submodule fork/commit

Keeps the submodule pointer itself always resolvable to a real upstream
commit (no acestep-only commits that would need to live on a fork,
diverge from upstream, or complicate `git submodule update`), while still
letting `ace-train` use a couple of small, genuinely necessary additions
to GGML that have no equivalent achievable through its public API alone.

## Current patches

- **`ggml-bf16-rtne.patch`** — adds `GGML_UNARY_OP_BF16_RTNE`: forward
  rounds an F32 tensor to BF16 precision (round-to-nearest-even) and back,
  staying F32-typed the whole time; backward is a straight-through
  estimator (gradient passes through unchanged). Used by `ace-train`
  (`src/dit-train-graph.h`) to match Python's bf16-autocast training
  precision on the LoRA branch's matmuls. Purely additive — a new op,
  zero changes to any existing op's behavior — and gated off by default in
  our own code (`g_dit_train_bf16_lora`), so nothing outside `ace-train`
  is affected even when the patch is applied.

  Why this needs a real GGML op and can't be done from application code
  alone: GGML's op dispatch (both the CUDA/CPU forward kernel selection
  and the backward-gradient formula selection) is a hardcoded
  `switch (tensor->op)` compiled into `ggml.c`/`ggml-cuda.cu` — there is no
  per-op plugin/extension point, only a per-*backend* one. Two
  submodule-free alternatives were built and measured before concluding
  this: (1) `ggml_map_custom1` (GGML's public escape hatch for arbitrary
  host callbacks) has no CUDA kernel and no backward case anywhere in this
  fork, so it needs `ggml_backend_sched`'s CPU fallback — correct, but
  ~2x slower per training step. (2) Two chained `ggml_cast` calls
  (F32->BF16->F32, genuine native ops) plus a transient spoof of the
  intermediate tensor's `type` field to dodge GGML's BF16-gradient
  assertion — fully native, no CPU/scheduler, but needs ~625 extra graph
  nodes for full per-matmul fidelity, adding ~15-40% per-epoch overhead
  since the training graph is rebuilt from scratch every step. Both are
  documented in `TRAINING_DEV.md` for the record; neither is in the
  current code. The patch file gives the original op's exact behavior
  (native speed, correct backward, zero extra nodes) without a submodule
  commit.

## Regenerating a patch after editing the submodule locally

```sh
cd ggml
# ... make your edits ...
git diff > ../patches/your-patch-name.patch
git checkout -- .   # revert the submodule's working tree back to pristine
```

Then re-run `cmake` (or reconfigure) to verify the patch applies cleanly
from a pristine checkout.
