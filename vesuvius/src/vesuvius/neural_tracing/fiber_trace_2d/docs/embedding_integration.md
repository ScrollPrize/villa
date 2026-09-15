# Causal embedding integration experiments

All variants score each beam candidate from its parent's memory, then update only
the chosen child's independent state. They use interpolated unit embeddings from
the frozen inference cache. There are no anchors and no GT inputs during tracing.
Old EMA behavior remains the default; experiments opt in through trace flags.

## Variants

- **Cadenced EMA:** admit a new observation after at least S base voxels of actual
  traced distance. At admission use decay `2^(-elapsed/H)`. Score candidates between
  admissions too. S is a minimum interval, not a rounded step count.
- **Finite window:** average cosine distance to admitted observations still within
  W base voxels of the candidate. Expire at the candidate's position, not the parent.
  This is a hard history range, not an EMA half-life.
- **Learned gate:** multiply the EMA update fraction by
  `sigmoid(bias + presence_coef*(presence-.5) - distance_coef*cosine_distance)`.
  Even an almost-rejected observation consumes the admission interval.
- **Angular limit:** calculate the ordinary EMA update, then cap its angular motion
  at `cap_degrees * elapsed/8`. Spherical interpolation follows the shorter arc.
  Cancellation and exactly antipodal ambiguity retain the previous reference.
- **Dual EMA:** independently maintain short/long half-life memories. Mix their
  candidate distances with `sigmoid(bias + slope*(distance_long-distance_short))`.
  Both are continuously adapted; they are not a bank of selected anchors.

Each state resets on a trace restart. Finite-history capacity is validated to cover
the requested physical range without silent truncation. Window experiments in this
matrix use at most257 observations.

## Fitting matrix

EMA: S=8,16,32,64 crossed with starting H=64,128,256,512,1024 (20).
Window: S=8,16,32,64 crossed with W=256,512,1024,2048 (16).
Gate/angular/dual: three distinct initializations each at S=8 (9).
Total45 candidates,150 Adam updates each, vectorized by family, exclusive managed GPU.
Half-lives and nonlinear coefficients are fitted; spacing/window are discrete.

The same30fit/9selection/11heldout whole-fiber groups are retained. Shared warm-up
is extended to4096basevoxels while keeping original branch locations and continuations.
Only fit loss produces gradients; selection loss chooses iteration and family
finalist. Heldout metrics are reported, never used for selection. Warm-up state is
recomputed at current parameters and detached before branch backpropagation.

The objective remains the existing native base-score plus linear weighted cosine
penalty, paired ranking loss, successful-route control loss and parameter regularizer.
This is a truncated fixed-path surrogate, not a differentiable beam-search rollout.
Each family's finalist must therefore receive a full50fiber native tracing eval.

## Running

Use the existing approved runner. `CONFIG` below is
`vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_fit_20260915_v1.json`.

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-extract --config CONFIG
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py integration-fit --config CONFIG
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py integration-audit --config CONFIG
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py integration-report --config CONFIG
```

Fit checkpoints include dataset, implementation/shared-fit hashes, complete matrix
and iteration count. Full tracing uses `prepare`, `validate-full`, then `full` with
each `eval_embedding_integration_{family}_20260915_v1.json` config.
Do not alter the frozen model roles, cache namespaces or evaluation thresholds.

Verification covers old default exact arithmetic, causal scoring, detached/padded
warm-up clocks, finite-window expiry, angular cancellation/antipodes, independent
dual memories, gradients and native/GPU real-vector parity. Torch uses a protected
acos derivative near endpoints; parity is checked to2e-5 absolute weighted score
on real exported routes, and2e-6 on synthetic cases. No model precision is changed.
