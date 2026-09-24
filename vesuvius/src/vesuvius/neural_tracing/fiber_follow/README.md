# fiber_follow

Supervised autoregressive tracing from a high-presence seed. A spatial network
proposes several continuations, scores them against image features and trace
history, and estimates how much of each continuation is safe to commit. It
traces both directions for inference and writes VC3D fiber JSON.

## Ground truth

**Every line point between the first and last control points of every fiber is
valid ground truth.** Line points and control points have the same authority.
The `reviewed` tag has no meaning for loading, sampling, weighting, losses, or
DAgger. Interpolation provenance does not change supervision either.

The loader preserves every interior line vertex and subdivides longer segments
to at most one trace-grid voxel. It excludes exterior tails and fibers with
fewer than two controls. All dense geometry, including low-presence regions,
trains point placement. Presence selects strong starting seeds; it supplies
neither pseudo-labels nor stop targets.

An outer annotation boundary is unknown continuation, unless explicitly marked
`kollesis_termination`. Missing targets beyond unknown boundaries are censored.
Tagged physical endpoints supply negative continuation labels. Control/span
metadata and source hashes are retained for reproducibility, without confidence
weights. Geometry identities invalidate stale seed and replay caches.

Coordinates are trace-grid voxels: one voxel is eight base voxels on this dataset.
The spatial split uses the arclength-weighted centroid of the controlled curve.
After augmentation, every training state must exclude the held-out z band from
its crop read block, active history, and dense targets, with an additional
48-voxel position guard. The same filter applies to replay and collection.

## Architecture and supervision

`spatial_candidates_v3` uses a 3D encoder-decoder with spatial skip connections,
metric coordinate channels, and early conditioning on 128 voxels of trace
history. The default crop is uniformly sampled **64×64×64**, with one trace-grid
voxel spacing on every axis. It covers 16 voxels behind and 47 ahead, and ±31.5
laterally. The current point stays offset toward the back of the oriented cube.
There are 16 future planes, two forward voxels apart; each has a 61×61 heatmap
covering ±30 lateral voxels with one-voxel spacing. This leaves image context
around every proposed point. The full-prefix head and 0.7 threshold are retained.

1. Plane heatmaps learn the dense GT curve crossings directly. Peak decoding
   connects modes into four coherent candidate paths with a curvature penalty
   and path-diversity suppression. This is local proposal decoding; rollout is
   greedy and has no multi-step beam yet.
2. The scorer samples decoded spatial features along each candidate and its
   lateral neighborhood. It predicts candidate rank and confidence for every
   prefix. A GRU carries information from the beginning of each candidate into
   every later confidence prediction; its state resets for each candidate and
   decision. Training scores current proposals, GT/jittered proposals, and the
   original proposals saved in replay.
3. Confidence labels compare interpolated candidates with dense GT crossings
   every 0.5 forward voxel by default. A prefix is positive when every known
   crossing agrees within `--tolerance` (default 1.5 voxels). A known mistake
   makes that prefix and subsequent prefixes negative; unknown continuation is
   masked. Already departed states supply negative confidence labels and no
   position targets. Prefix checking begins at the first predicted point, so a
   perturbed state can learn to recover onto its fiber.

Inference selects the highest-ranked candidate whose first prefix clears the
confidence threshold (default **0.7**), then commits up to four safe points (2–8 forward voxels).
Confidence is made nonincreasing along each candidate. If no candidate clears
the threshold, it stops **before** committing. There is no stop trimming or
presence-based model-stop heuristic. Bounds, loops, and maximum length remain
geometric limits. Heading updates use committed points only.

The 0.7 default is shared by training diagnostics, DAgger collection, and
inference, and can be overridden with `--confidence`. It was selected from
short held-out rollouts of the preceding model; recalibrate it after training
the new head.

The confidence threshold is an operating parameter, not a demonstrated
calibration guarantee. Compare coverage and wrong length on held-out rollouts.
Logs separate best-proposal error/recall from selected-proposal error to help
distinguish proposal failures from scoring failures. Error diagnostics clip
lateral errors at eight voxels and exclude GT/jitter/replay candidates.

`target_crop_oob`, `target_crop_edge`, and `target_heatmap_oob` report fractions
of known dense crossings outside the crop, within three voxels of its lateral
edge (including outside), and outside prediction support. Lower is better.
These measure sample geometry, not model accuracy; no GT is discarded from
confidence supervision because of these counters. Position heatmap loss only
applies where the target is representable.

Batch images mark the current point and prediction limits in cyan, supplied
history in red, GT in green, and the full proposed continuation in orange.
Magenta squares flag GT outside the crop. Plot bounds remain the actual crop;
orange shows the full proposal before confidence gating. Rollout images show
the model-generated path starting from a single seed.

## CT-only, native level-0 experiment

Use `scripts/launch_ct_tube.sh RUN_NAME` for a fresh CT-only Gaussian-tube run.
The preset uses `/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/0`.
One CT voxel in this dataset is four base voxels (`--ct-grid-scale 4`). All trace
geometry and tolerances remain in the original eight-base-voxel trace grid.
The oriented crop is 64 cubed with `--crop-spacing 0.5`: one native CT voxel per
sample, half the previous physical field of view. This reads the level-0 voxels
directly and samples that resolution, rather than reducing CT to the old grid.
Forward planes are one trace voxel apart and lateral heatmap samples are half a
trace voxel apart. The 16-plane horizon is now 16 trace voxels (32 CT voxels).

The two image channels are CT intensity / 255 and the tracer's history tube.
Metric coordinate channels and geometric history conditioning remain. Presence
is used for initial seed selection/snapping and a local weighted-PCA heading;
CT mode never opens the predicted nx/ny arrays. Inference also accepts an
explicit `--heading x,y,z` for each seed. After initialization, crop reads and
rollout images use CT. Presence-based break statistics in evaluation are only
additional diagnostics, not inputs or stopping rules.

`--heatmap-target tube --tube-sigma 0.35` supervises a full 64-cubed scalar field
with `exp(-distance_to_annotated_polyline**2 / (2*sigma**2))`, truncated at four
sigma. Sigma 0.35 trace voxels equals 0.7 native CT voxels. Distance is measured
to original line segments, retaining bends and multiple plane crossings. This
is different from the previous per-plane normalized Gaussian crossing targets.
Unknown annotation ends censor outward continuation; physical endpoints retain
background supervision. Already-departed states retain confidence negatives
but have no tube-position loss. The annotated tube is a target, never an input.
The model still receives only its own history at rollout time.

The tube uses sigmoid outputs and squared-error regression, averaging losses
in the three-sigma tube neighborhood and remaining known background separately
to keep empty voxels from dominating. Existing proposal decoding, candidate
ranking, prefix confidence and DAgger remain. The decoder samples the volume on
forward planes, so its forward-monotone path representation is unchanged even
though the supervised volume can represent arbitrary annotated bends.

`images/tube_STEP.png` shows image, history, target tube, predicted tube, and
known-region masks in two orthogonal projections. `batch_STEP.png` now shows
CT behind the proposed path. Tube checkpoints record target mode, sigma, CT
voxel scale and crop/heatmap spacing. Start fresh; old fiber-input checkpoints
and replay do not match this configuration. Run names remain unique.

```bash
bash src/vesuvius/neural_tracing/fiber_follow/scripts/launch_ct_tube.sh ct0_tube_64
# Override ordinary training settings after the name if desired:
# ... ct0_tube_64 --steps 10000 --batch 16

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/neural_tracing/test_fiber_follow.py \
  tests/neural_tracing/test_fiber_follow_ct_tube.py
```

## GPU execution

CUDA training and checkpoint loading use channels-last storage for 3D
convolutions. The tensor dimensions, weights, losses, batch size, and BF16
training precision are unchanged. Fixed-size training batches enable cuDNN
kernel tuning; diagnostic rollouts disable tuning because active batch sizes
change. Independent inference and collection do not enable tuning. CPU execution
keeps its regular tensor layout.

The selected kernels can introduce floating-point rounding differences and
change choices between nearly tied proposals. Fixed seeds do not imply a
bit-identical optimization trajectory across layouts or kernel choices. Run
configuration records `cuda_channels_last_3d` and `cudnn_benchmark_training`.
See `EXPERIMENTS.md` for timing and numerical checks.

## Continuous DAgger

Training keeps one optimizer running. Every `--dagger-every` steps (default 500),
if the collector is idle, it saves a policy snapshot and starts a background
collection process. It does not wait for collection. Completed caches are
published atomically; persistent loader workers refresh replay every eight
chunks, with normal loader prefetch delay. A busy collector skips that snapshot
opportunity and launches at a later interval.

Collection uses high-presence seeds, records the exact input frame and actual
trace history before each decision, and matches progress against the seed's
original fiber. It marks the preceding 48 voxels as hard near departure or a
would-stop decision, retains up to 24 voxels after departure, and optionally
explores eight additional calls after a confidence stop. Exploratory states are
explicitly marked. Unknown endpoint crossings and held-out space censor
collection. Ordinary decisions are thinned to 16-voxel spacing; each trace
contributes at most 192 states.

With both replay pools available, sampling targets 50% fresh perturbed states,
30% ordinary replay, and 20% hard replay. Missing pools fall back to fresh data.
The latest four completed collections are active by default; newer collections
receive linearly greater sampling weight. Each archive records its checkpoint,
training step, collection settings, crop, volume, and fiber identities. Logs
report actual sample fractions and cumulative replay samples consumed. Older
archives remain on disk for auditing or explicit reuse.

`--dagger-device` can place collection on another GPU or on CPU; its default is
the training device, where the two processes share compute and memory. Start
with a small `--dagger-batch` if memory is constrained. Asynchronous completion
changes the exact sample ordering across runs. Use recorded fixed caches with
`--dagger-every 0` for controlled replay experiments. Training shutdown stops an
unfinished collector and reports it; incomplete collections are never published.

## Commands

From `vesuvius/`, use the existing environment. The codec may need:

```bash
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
FF=src/vesuvius/neural_tracing/fiber_follow

# Fresh spatial model; collection and replay run automatically during training.
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.train \
  --name spatial_dagger_v3 --steps 10000 --batch 32 \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers

# Fixed geometry-bound validation seeds. Rebuild after annotation geometry changes.
.venv/bin/python "$FF/scripts/eval_ckpt.py" field --seeds-only --rebuild-seeds
.venv/bin/python "$FF/scripts/eval_ckpt.py" \
  "$FF/output/spatial_dagger_v3/last.pt" --tag spatial_dagger_v3 \
  --params '{"confidence":0.7,"n_commit":4}'

# Independent collection is also available, e.g. for fixed replay ablations.
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.collect \
  --checkpoint "$FF/output/spatial_dagger_v3/last.pt" \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
  --max-seeds 64 --out /tmp/fiber_decisions.npz

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/neural_tracing/test_fiber_follow.py
```

Add `--dagger-every 0` for fresh-only training. Add `--onpolicy CACHE...` (oldest
to newest) for existing replay. `--init` accepts only this exact architecture and
configuration and starts a new optimizer. `--allow-history-change` permits only
the history renderer and sigma to differ when initializing weights. The uniform-crop v3 architecture requires a fresh
run; earlier checkpoints cannot initialize it. Continuous DAgger itself never
restarts the optimizer. Run names must be new. There are no old-checkpoint
adapters, history-padding fallbacks, or cache migrations.

Each run contains `config.json`, `log.jsonl`, checkpoints, optional diagnostic
images, and `dagger/` with source snapshots, decision archives, memory-mapped
arrays, collection logs, and the atomic `replay.json` index. Replay format is v3;
the controlled-span evaluation format remains v2, with geometry-bound identity.

Evaluation conserves `correct + offtrack + unknown == length`. Read scored
`length_precision` together with coverage, `unknown_length_fraction`, and
`verified_length_fraction`; unknown tails are never credited as verified.
Historical experiments in `EXPERIMENTS.md` are not evidence for the quality of
this newly implemented architecture. It needs a full training/evaluation run.


### Smooth history for the CT tube experiment

`bash scripts/launch_ct_tube.sh RUN_NAME` now uses connected Gaussian history
segments (including the last segment to the current position), sigma 0.35 trace
voxels = 0.7 native CT voxels, and zero independent point jitter. Smooth drift
and wobble remain. Empty histories stay empty and masked gaps are not joined.
The renderer and width are saved in each checkpoint and used in rollout and
replay collection. Old checkpoints retain their original point rendering.

To initialize this experiment from a previous CT run, stop it with
`bash scripts/stop.sh OLD_NAME`, then use a new run name and add
`--init output/OLD_NAME/ckpt_NNNNNN.pt --allow-history-change` to the CT launcher.
Use a completed numbered checkpoint. This transfers weights, starts a new
optimizer and step counter, and collects new replay; it is not an exact resume.
`stop.sh` checks the recorded training PID and terminates only that run's process
tree (including forkserver workers), rather than matching all Python workers.

History regressions can be checked with the existing environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/neural_tracing/test_fiber_follow.py \
  tests/neural_tracing/test_fiber_follow_ct_tube.py \
  tests/neural_tracing/test_fiber_follow_history.py
```

Batch diagnostics label fresh versus replay samples, mark already-departed
examples as `OFF TRACK: reject continuation`, and show the selected proposal's
next-step confidence. In tube diagnostics these examples retain the reference
geometry for inspection but explicitly mark it as masked: only rejection
confidence is supervised, not tube position or ranking.
