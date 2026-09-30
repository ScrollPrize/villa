# Continuous candidate memory (v5)

`--memory-version 5` adds an optional architecture, `axial_fiber_memory_v5`.
Existing v2/v3/v4 defaults and checkpoint loading remain supported. V5 requires
feature-memory revision 2 and `--no-correction`; its refinement is the shared
trajectory decoder, not the old local correction head.

## Model

The main eight-channel crop is encoded once and revision-2 memory is observed
once. The seed, 16 slots, 64-observation cache, detail tokens, admission logic,
and memory-conditioned deep features are inherited from v4 revision 2.

At each of 16 forward planes, a small shared head samples fine and conditioned
deep features over a 97 x 97 lateral map at 0.5 trace-voxel spacing (150,544 total
locations for the standard crop). It predicts a score and two offsets bounded
to half the actual sampling spacing. The positions used by route selection are
continuous from the outset. The grid is not an output-coordinate constraint.

Greedy suppression operates on these continuous positions with radius 0.5,
retaining up to 32 nodes per plane. Invalid/padded nodes have explicit masks.
The first plane is filtered by the existing recovery distance before selection.
Ranking heads and route scores use FP32 to avoid BF16 quantization ties.
Tie-breaking uses the first flattened argmax. Route node scores are the dense
proposal log probabilities at the retained nodes. The historical memory reaches
the route only through the memory-conditioned deep features and, separately,
the confidence head; no per-candidate memory read exists.

A second-order max-sum dynamic program connects candidates across all adjacent
planes. There is no grid adjacency or hard turn limit. Soft costs use unsigned
direction-tensor alignment (weight 0.1) and Huber changes of unit segment tangent
(weight 0.05). Tensor trace and anisotropy reduce the influence of missing or
ambiguous directions. A disconnected route has an explicit flag and zero
returned continuation confidence.

The selected continuous curve initializes the existing trajectory decoder.
Two refinement passes read the image and historical memory, refreshing local
evidence and sharing per-layer memory projections. Each pass has the existing
one-voxel displacement norm bound, crop bound, and first-connection bound. There
is no clamp back to the original proposal region. No candidate or refinement
is written into observation memory.

## Objectives and sampling

Annotation crossings supervise dense proposal cross-entropy with bilinear
weights. Only lattice sites whose half-cell contains a target learn its exact
offset; boundary ties are averaged. Both losses precede shortlisting, so a
missed target still receives a training signal. Original-fiber localization is
supervised in departed states when identity and the crossing are observable.
Missing/censored crossings and unobservable identity remain masked.

Coverage is a diagnostic only: a plane is covered when its shortlist holds a
node within one proposal spacing of the annotated crossing. No ground truth
candidates are injected into the model's route.

The two new objectives start with weight 1 each. Existing dense geometry
(75% final, 25% mean earlier proposals), prefix confidence, candidate confidence,
visible-reference identity, and memory probes remain. Geometry gradients reach
selected offsets; the dense classification loss trains the discrete proposal
choice. This is not differentiable dynamic programming or structured route
likelihood training.

V5 uses v4's causal two-decision chunks, 128-observation streams and selective
endpoint replay. All three new objectives participate in endpoint replay.
The existing stale-feature approximation remains: three age-stratified crops
are re-encoded, other stored features are detached, and every writer transition
is replayed at current weights. Replay weight remains 0.5.

JSON `memory` metrics include proposal coverage, nearest-candidate error,
initial/final curve error, disconnected counts, and errors grouped by annotated
turn angle (<10, 10–30, >=30 degrees). Sum errors/counts before computing means.
Coverage uses a 0.5-voxel radius at the default spacing. These are sampled-state
diagnostics, not tracing accuracy or held-out generalization results.

## Launch and initialization

From `fiber_follow`, this launches a **new** background run:

```bash
RUN_NAME=axial_candidate_memory_v5_run1 bash scripts/launch_candidate_memory.sh \
  --init-tracer output/axial_feature_memory_v4_revision2_run1/ckpt_015000.pt
```

The launcher sets the full configuration: 32 candidates/plane, 0.5 spacing,
two refinement passes, revision-2 memory, batch 8 and microbatch 4. Omitting
`--init-tracer` starts randomly initialized training. The existing launch script
rejects occupied run names. Creating this architecture does not automatically
launch a long training job.

V4 revision-2 initialization strictly copies compatible EMA weights, replaces
the direct coordinate head, and initializes the proposal head.
The first refinement-stage embedding is retained and added stages start at
zero. The shared displacement head is retained, so this is a behavior change.
The optimizer and schedule start fresh. For transfer initialization, inherited
weights are frozen for the first 500 updates, then receive 0.1 times the new
heads' learning rate. `--proposal-warmup-steps` and
`--proposal-inherited-lr-scale` configure these settings. Optimizer group metadata
preserves them through subsequent v5 resumes. Random initialization uses a
single ordinary optimizer group without this transfer warmup.

To continue an existing run with a fresh optimizer, use its existing launch
arguments with `--resume <checkpoint> --reset-optimizer` and no `--init-tracer`.
This retains raw model weights, EMA, RNG, replay banks and global update count,
but starts empty AdamW state with one LR for every parameter and no transfer
freeze. The usual LR warmup restarts, followed by cosine decay over the remaining
updates. Checkpoints store `lr_restart_step` so subsequent ordinary resumes
continue that schedule. Omit `--reset-optimizer` on those later resumes; supplying
it again explicitly requests another reset. The original checkpoint is unchanged.

For direct Python construction, set `memory_version=5`, `memory_slots=16`,
`correction=False`, `feature_memory_revision=2`, and
`recurrent_refinement_steps=2` in `DirectConfig`, then call `build_model`.
The shared dataclass retains legacy defaults; the v5 launcher supplies the
v5-specific ones explicitly.

## Removed shortlist identity head (2026-09-29)

The first v5 design added a per-candidate identity head: each retained node
queried the historical memory through cross-attention and its logit was added
to the proposal log probability before route search, trained by a Gaussian
cross-entropy toward the annotated crossing over the shortlist. In
`axial_candidate_memory_v5_run1` (stopped at step 7000) a CPU replay of the
captured real-sample fixture (63 decisions, 1008 supervised planes, identical
causal memory) showed the head changed the per-plane winner on 26 planes
(2.6%), moved it closer on 17 and farther on 9, and crossed the 1.5-voxel line
twice in its favour and never against; its argmax agreed with the proposal
argmax on 90% of planes, it placed 28% of its mass on positive nodes with
logits near 10, and its cross-entropy sat at 2.35 against a 0.53 floor. Its
label was a localization target that appearance alone answers, so it received
no pressure to use memory. The head, its loss, `proposal_identity_weight`, and
the `proposal_identity_logits` output were removed. Checkpoints from run1 hold
`candidate_*` weights and no longer load strictly; new runs initialize from a
v4 revision-2 checkpoint as before. Fixture-paired errors from that replay were
0.700 voxels for the v4 initialization and 0.780 for v5 at step 7000, so the
remaining architecture has not yet beaten its initialization.

## Paired real-data cost benchmark

The harness captures real augmented samples with original-fiber labels and
complete causal histories. It freezes the replay index and hashes the crops and
source checkpoint. Capture once, then run each version in its own process:

```bash
export PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src
export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=4
PYTHON=/home/sean/Documents/villa4/vesuvius/.venv/bin/python

"$PYTHON" -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_candidate_memory capture \
  --checkpoint output/axial_feature_memory_v4_revision2_run1/ckpt_015000.pt \
  --out output/candidate_memory_v5_validation/real_samples --streams 4

"$PYTHON" -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_candidate_memory run \
  --fixture output/candidate_memory_v5_validation/real_samples \
  --version 4 --profile --warmup 8 --repeats 20 \
  --out output/candidate_memory_v5_validation/v4.json

"$PYTHON" -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_candidate_memory run \
  --fixture output/candidate_memory_v5_validation/real_samples \
  --version 5 --profile --warmup 8 --repeats 20 \
  --out output/candidate_memory_v5_validation/v5.json
```

Training uses compiled BF16 with FP32 memory, batch 8, microbatch 4, two traces
and two decisions per chunk. All weights participate in backward; transfer
freezing is disabled for cost measurement. LR zero keeps both model workloads
stationary while still running AdamW, clipping, and EMA. Inference also uses the
compiled forward, batch one, and actual reconstructed historical memory. Pass
`--no-compile` only for an explicit eager comparison. Warmup/compilation are
excluded and measured recompilation causes an error. GPU allocation is capped
per process (default 80%).

Ordinary updates exclude endpoint replay. The replay mode adds one real archived
endpoint per update and records its actual historical length; it measures replay
event cost, not the average replay frequency of a full run. H2D transfer is
included in training; volume loading, collection and diagnostics are excluded.
Input files are identical between versions. New v5 heads are untrained, so these
measurements establish cost and integration, not accuracy or convergence.

### Measured results (2026-09-29)

RTX 5090, PyTorch 2.12.1+cu130; the commands above used 8 warmup iterations and
20 measured iterations per mode. The captured fixture contains 85 real crop
decisions from four completed histories (lengths 5, 18, 18, 44). Eight full
two-trace windows were selected evenly through the capture and reused between
architectures. Both training comparisons below use `torch.compile`.

| Workload | V4 mean / p50 / p95 (ms/update) | V5 mean / p50 / p95 (ms/update) | V4 crops/s | V5 crops/s |
|---|---:|---:|---:|---:|
| Ordinary update, 8 crops | 439.9 / 439.6 / 452.0 | 500.0 / 501.6 / 519.7 | 18.19 | 16.00 |
| 8 crops + one endpoint replay | 817.3 / 867.1 / 950.9 | 889.8 / 923.3 / 1021.7 | 9.79 | 8.99 |

Ordinary update time increases 13.7% (throughput decreases 12.0%). With one
endpoint replay per update, time increases 8.9% (throughput decreases 8.1%).
Replay rows count ordinary crops in the throughput numerator; they are not
estimates of a full run's average replay frequency. Peak allocated training
memory is 12.73 GiB for v4 and 13.89 GiB for v5; peak reserved memory reaches
13.65 and 15.34 GiB respectively.

V5 compiled batch-one inference is 28.0 ms mean, 27.6 ms p50, 29.4 ms p95
(35.72 decisions/s). No v4 inference comparison is needed for this report.
The original `v4.json` inference row used eager execution and is excluded;
its training and replay rows were compiled. Current harness defaults compile
all three modes. Raw timings and profiles are in
`output/candidate_memory_v5_validation/{v4,v5}.json` and the adjacent
`.profile.txt` files. `real_samples/capture.json` records input/checkpoint hashes.

One unmeasured profiled ordinary update per version shows convolution backward
remains the largest operator (v4 92.6 ms, v5 90.7 ms self CUDA time). cuDNN
attention forward grows from 43.4 to 65.6 ms, consistent with the extra decoder
pass; other candidate attention, sampling, ranking and route work is included
in the total measured cost. Profiling is separate from the throughput timing.

## Tests

`tests/test_candidate_memory.py` covers route search against brute-force
enumeration, curved routes, sign-invariant directions, suppression/padding,
continuous initial coordinates, departed-state localization gradients through
old observations, masking, migration/checkpoints, endpoint replay, causal data
construction, cold/streaming tracing, transfer warmup and compiled gradients.
Run it alongside the existing trajectory, detailed-memory and refinement suites.

Validation before the identity head was removed: 528 tests passed, 23 skipped,
2 deselected, and 10 subtests passed in the CPU suite. Separate CUDA tests passed
for v4 revision-2 compiled BF16 gradients and v5 compiled proposal/memory
gradients. See the removal note above for the re-run after that change. The v5 check verifies
the same selected routes, gradient cosine similarity above 0.99, and gradient
norm agreement within 5%. New ranking heads use FP32 because BF16 score ties
otherwise made eager and compiled candidate selection unstable. Existing
encoder/decoder precision is unchanged. Test logs are saved alongside the raw
benchmark results.
