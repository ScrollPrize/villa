# Learned costs inside the VC3D beam

`beam_step_cost_v3` replaces candidate scoring while keeping the native cone
proposals, beam search, endpoint constraints, bidirectional fusion and restart
metric. It does not predict coordinates or a dense fiber tube. Old
`beam_rerank_v2` checkpoints are rejected; this architecture trains from scratch.

## Search and model

With a hook, C++ exposes **every valid cone proposal at every generation** before
hand-ranked parent caps, pool selection, endpoint diversity pruning, or target
acceptance. Defaults retain 32 intermediate branches using learned scores, then
8 after the second lookahead step. This is bounded learned lookahead rather than
exhaustive two-step enumeration. The 81-direction cone therefore supplies up to
648 proposals from an eight-wide beam and 2,592 from the intermediate beam.
Invalid prediction samples remain excluded. Without a hook, the original native
search is unchanged, including its hand-scored lazy lookahead.

The model uses **CT level 1 only**, at 8 base voxels per sample. Its default
192 × 96 × 96 crop covers 128 trace-grid voxels behind the current frontier,
63 ahead, and ±47.5 laterally. Inputs are CT and rendered observed history.
The encoder also conditions on 128 history voxels sampled into 32 geometry
observations. Each candidate supplies its own most recent 32 voxels of path,
its parent endpoint and its proposed endpoint. A pooled spatial context exposes
the whole crop, including forward context, to the path scorer.

A shared encoder runs once per crop, then a small path network scores proposals
in chunks of 256. Chunking does not discard candidates. If surviving branches
spread beyond one crop, additional crops at the **same resolution** cover them;
all scores retain their original proposal correspondence.

The output is one on-fiber logit. Its nonnegative step cost is
`step_length * softplus(-logit)`, where length is in trace-grid voxels. C++ carries
`parent_cost + step_cost`. The hand cost of the proposed step is excluded, and
old segments are not charged again. Optional confidence stopping applies to
open-ended traces. Target-reaching candidates are also scored before acceptance.

## CT loading

Beam and direct now call the same `crop_sampling.scalar_crops` helper: the
`read_tight_blocks` reader plus fused Numba interpolation/history rendering.
It reads the smallest axis-aligned block enclosing each oriented crop rather
than the older rotation-invariant cube. `ChunkedArray` memory-maps uncompressed
Zarr chunks. The existing local `s1_ds2.zarr/1/.zarray` has `compressor: null`;
no new decoding or data rewriting is required. This model uses the direct
follower's 3D crop path.

## Training

Bootstrap training runs the native search with an observing hook and hand costs,
including perturbed starts and mined failed spans. Pools are processed as they
arrive and retained in bounded reservoirs. Up to 256 uniformly sampled proposals
per state bound training memory; this is **not** an inference prefilter. Labels
check the newly proposed segment against the original annotated fiber, including
backward progress and censored unknown endings. BCE and listwise quality ranking
train the same logit used at inference. Wholly censored pools contribute zero
loss without introducing NaNs. There are no tube or prefix heads.

The holdout guard still checks the conservative crop/history footprint and all
candidate paths before a state enters the training reservoir. Bootstrap states
come from hand-guided search; model-driven collection remains a potential next
step if closed-loop evaluation exposes a distribution gap.

## Build and run

From the monorepo root, with the existing CMake Python build:

```bash
ninja -C volume-cartographer/build-dev -j 4 vc_fiber_trace VC3D vc_fiber_trace_metric
export VC_PYTHON_BUILD_DIR="$PWD/volume-cartographer/build-dev"
```

`VC_PYTHON_BUILD_DIR` selects that build for training, inference and forkserver
workers without reinstalling packages. Otherwise an installed binding exposing
`learned_scoring`, `parent_losses`, and `step_lengths` is required.

From `fiber_follow/`:

```bash
bash scripts/launch_beam.sh beam_v3 --steps 20000 --batch 2
```

Training throughput notes: the CUDA training forward is `torch.compile`d by
default (`--no-compile` disables it; diagnostics stay eager), the model uses
NCDHW rather than channels-last, AdamW is fused, and per-step metrics are kept
on the GPU until a log step. Hook pools carry each distinct parent path once
plus one endpoint per candidate (`parent_path_points`, `parent_path_offsets`,
`parent_index`, `endpoints`); `paths()`/`path_points` still rebuild full paths
on request. Hard-span mining runs `--mine-workers` (default 16) single-threaded
forkserver processes, one fiber each; results equal in-process mining.

Default batch is two full crops. `--pool-size` controls only training proposal
sampling; `--score-chunk` controls scoring memory; `--lookahead-width` controls
intermediate native beam width. The old `--hook-mode`, `--hook-weight`,
`--hook-every-rounds`, `--k-fwd`, and tube/prefix options are removed.

Evaluate using `beam.evaluate_spans` and `scripts/eval_ckpt.py`, or re-trace
control-point spans using `beam.infer`, as before. These Python entrypoints use
the native VC3D tracer; the GUI does not load a PyTorch checkpoint automatically.

CPU regression coverage includes all-proposal access despite tiny legacy caps,
learned scoring before target acceptance, vetoed targets, incremental cost
accounting, live-frontier geometry, separated branches, censored losses,
forward-context gradients, chunk equivalence, training/checkpoint/inference,
holdout exclusion, mining, and native no-hook parity. Synthetic smoke runs do
not establish tracing accuracy or GPU memory/latency at the full defaults.

To compare real level-1 crop loading against the previous reader:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  ../../../../.venv/bin/python scripts/benchmark_beam_inputs.py \
  --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs --iterations 12
```

The harness checks bit-identical CT/history values for three orientations before
reporting warm-cache mean, median and p95 crop times and source voxel counts.

Recorded local result (`input_benchmark.json`): 12 warmed calls, one CPU thread,
three orientations, the default full crop, and a shared 256 MiB reader budget.
Cubic reader mean / p50 / p95: **159.7 / 37.5 / 611.1 ms**; shared tight reader:
**92.5 / 29.3 / 331.0 ms**. All CT/history values were bit-identical, and the
reader retained 27 memory-mapped chunks. These orientation/cache-sensitive
measurements describe CPU preparation only, not training or tracing throughput.

Validation in this workspace: 122 Python checks passed and one CUDA check was
skipped across the beam, direct, sampler, single-path and binding suites; the
native review and 3D suites passed 9 and 51 cases. The beam suite includes a
two-update CPU training run and model-in-the-loop tracing. No production training
run was started. Tests used the existing project Python 3.14 environment and
pytest from the already installed `spiral-fitting` environment (no installs).

A separate one-update synthetic smoke also passed with a forkserver worker and
reloaded its checkpoint using the build-directory binding override. The sampler
and native readers were opened inside the worker, as in the training launcher.

The native Python binding, VC3D application, and `vc_fiber_trace_metric` also
rebuilt successfully in the existing `RelWithDebInfo` build tree using Ninja.

## Training diagnostics and terminal output

The trainer uses the shared readable terminal formatter, including ranking and
on-fiber losses, candidate correctness, span success/restarts, and rollout
coverage/precision. `log.jsonl` retains the complete numeric records.

Every `--diag-every` updates (default 500), `images/` receives the candidate-pool
image plus `rollout_000500_hand.png` and `rollout_000500_model.png`. The paired
open-ended traces use fixed held-out seeds, with CT backgrounds, green GT,
orange traces, stop reasons and followed/available lengths in two straightened
views. These traces have no control-point resets. They are scored once and the
same paths are plotted. `--diag-max-len` defaults to 400 trace-grid voxels and
`--diag-confidence` to 0.5 for learned rollouts; span diagnostics remain ungated.
One seed position per diagnostic fiber is selected once (both directions), using
the shared presence-supported seed selector. If there are no eligible held-out
seeds, the log explicitly reports that rollout images were skipped.
