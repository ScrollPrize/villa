# Reviewing automatically traced neighbor fibers

`neighbor_mining.py` generates proposed negatives offline using the native VC3D
fiber tracer. It does not modify training labels. The first review batch is in
`output/neighbor_negatives/index.html`: 12 candidates with an individual PNG and
JSON for each. Cyan dashed lines identify the annotated target; orange lines
identify the proposed negative; diamonds mark the two initialization points.
Both large panels show the same target–negative pair in the same straightened
coordinates, with CT above and presence below. Direct labels identify the two
paths. A small third panel shows their actual 3D separation; an apparent overlap
in a projection does not imply a 3D intersection. Backgrounds are maximum
projections through a thin slab whose thickness is labeled, not single slices.

The JSON includes both polylines, the seed points, annotation source hash,
coordinate scale, validation measurements, and `status: pending_review`.
`report.json` records all parameters and timings; `attempts.json` records trace
rejections. Output directories must be empty, preventing accidental replacement
of an existing review batch.

## Generation

From this `fiber_follow` directory, using the existing configured native build:

```bash
AGENTS_AGENT_MODE=1 ninja -C /home/sean/Documents/villa4/volume-cartographer/build-dev vc_fiber_trace -j 8

AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fiber-review-matplotlib \
/home/sean/Documents/villa4/vesuvius/.venv/bin/python -m \
  vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
  --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/1 \
  --ct-grid-scale 8 \
  --native-build-python /home/sean/Documents/villa4/volume-cartographer/build-dev/python \
  --seed-spacing 20 --extrapolation 10 --count 12 --max-anchors 160 \
  --output output/neighbor_negatives_next
```

No package installation is needed to use the existing build. The CLI requires
the presence/nx/ny uint8 level-3 stores to be decoded in place. The provided
direction stores have been decoded at levels 3 and 4 and verified against their
compressed originals. Missing direction bytes are excluded. Direction decoding
uses the existing native decoder; axes are compared without regard to sign.

The seed spacing and extrapolation lengths above are in fiber-grid voxels
(8 base voxels each). The helper `trace_controls` accepts two or more xyz
control points, traces the intervening segments, and extrapolates from both
ends. The low-level Python binding also exposes point-plus-direction
extrapolation. The native tracer's coordinate scale is checked explicitly;
this run uses four native trace voxels per fiber-grid voxel.

### Longer bank for 128-voxel wrong continuations

The bulk CLI now exposes `--seed-spacing`, `--extrapolation` (at each end), and
`--block-size`. Defaults remain 20/10/80 for short-bank generation. The new
`output/neighbor_negatives_bulk_tail128_v1` uses 20/70/192: approximately
160-voxel paths in a 192-voxel validation cube. This fits a 128-voxel wrong-fiber
tail, the 16–24 voxel bridge, and a four-voxel end margin. All lengths are in
trace-grid voxels. The same whole-path acceptance and holdout checks apply.

Launch using the existing environment and native build, without installation:

```bash
bash scripts/launch_neighbor_bank_128.sh --max-shards 6  # bounded pilot
bash scripts/launch_neighbor_bank_128.sh --resume       # after the pilot exits
tail -f output/logs/neighbor_negatives_bulk_tail128_v1.log
```

Omit `--max-shards` for a fresh full sweep. The launcher uses 12 workers on the
24-core host (override with `--workers N`), CPU
nice 19 and idle I/O priority where available. It records its PID and exact
command under `output/logs/`. `report.json` and `bank.json` publish progress;
`complete: false` banks can already be used by training. Use a new directory
when changing mining parameters; an existing bank's geometry and policy are
immutable. Longer paths may fail whole-path checks more often than short ones.

### Variable stored path lengths

Both the bulk and preview generators accept `--min-path-length` and
`--max-path-length`, in trace-grid voxels of **total stored path arclength**.
Supply both options. The maximum overrides `--extrapolation`; the block must
still be at least `max-path-length + 8` wide. Without the range options, fixed
extrapolation behavior is retained.

The producer traces toward the maximum, retaining partial native extrapolations
as proposals if they stop early. It validates the maximum available window
around the seed span. If that fails, it checks the minimum window and searches
between a passing minimum and failing maximum to within `sample_step` (0.25
voxels). Every returned window is independently validated by the same strict
whole-path checks; trimming preserves polyline corners. An unsafe minimum
window or a trace shorter than the minimum is rejected. This conservative
search does not exhaust every possible window position and does not make the
length distribution uniform.

The 80–160 bank uses a 192-voxel block and 12 workers:

```bash
bash scripts/launch_neighbor_bank_80_160.sh
# Resume this same bank after a stop:
bash scripts/launch_neighbor_bank_80_160.sh --resume
```

It writes `output/neighbor_negatives_bulk_80_160_v1`, with its own logs and PID.
The old approximately 160-voxel bank remains in
`output/neighbor_negatives_bulk_tail128_v1`. Its original generator modules are
preserved under `source_snapshot`; the launcher uses that source root when
present, so the old run can still resume with its recorded implementation
hashes. Changed mining policies always use a separate bank directory.

The wrong-continuation sampler still requests tails from 4–128 voxels. Each
selected path must fit the requested tail plus its 16–24 voxel transition and
four-voxel margin; an undersized draw is rejected, never silently shortened.
An 80-voxel path therefore supports tails up to 52–60 voxels, while a 160-voxel
path can support the full 128. Smaller paths also remain usable as local
identity negatives.

### Outer search band and duplicates across shards

The preview and bulk CLIs also accept `--min-distance` and `--max-distance`.
These bound the distance of the **entire retained path** from the continuous
target annotation, with the same conservative interpolation margins used for
separation checks. Seeds outside the band are excluded before tracing.
Defaults are 0 and 12, in addition to the separate 2.5-voxel target exclusion.

The new independent bank uses an inner radius of 12, an outer radius of 32,
80–160-voxel paths, a 192-voxel block and five workers at nice 19:

```bash
bash scripts/launch_neighbor_bank_outer.sh
# Resume after an interruption:
bash scripts/launch_neighbor_bank_outer.sh --resume
tail -f output/logs/neighbor_negatives_bulk_r12_32_l80_160_v1.log
```

Its output is `output/neighbor_negatives_bulk_r12_32_l80_160_v1`. It neither
loads nor compares against previous banks. New identity runs infer the
32-voxel sampling radius and read matching CT patches separately at positive
and negative query locations. The main fine crop stays unchanged.

Bulk workers write unpublished proposals under `pending/`. The coordinator
accepts them in annotation/anchor order, applying one spatial coverage index
across **all** shards and target annotations. Duplicate decisions therefore do
not depend on worker completion order. An ordinary following draw is suppressed if at least
80% of its uniformly sampled arclength lies within two voxels of previously
accepted geometry with axis alignment within 25 degrees. Reversed traces,
changes in vertex spacing, shifted rediscoveries and coverage split across
several older paths are handled. Crossings and mostly new extensions remain
eligible. This detects overlapping geometry, not the biological identity of
disconnected spans; parallel fibers closer than the geometric tolerance can
also be merged by this criterion for following draws only.

Bank format v2 keeps every validated `(parent, arc range, eligibility)`
relationship with its exact path, including overlapping rediscoveries. The
`draw_eligible` mask applies global deduplication only to ordinary following
sampling. Primary negatives and wrong continuations keep access to all certified
relationships; approximate overlap never substitutes another parent's geometry.
Evaluation paths do not suppress training draws. Shard metadata records training
and draw indices plus path lengths for length-aware selection.

Generate a fresh combined nearby/outer bank with:

```bash
bash scripts/launch_neighbor_bank_shared.sh
```

This writes `output/neighbor_samples_r0_32_l80_160_v2`, retains 80–160 voxel paths
up to 32 voxels away, and uses five low-priority workers. The normal 2.5-voxel
exclusion remains active. Start training once `run.json` exists; no completed
shards are required at startup. Version-1 banks remain readable but cannot
recover relationships discarded by their old producer. The changed format and
source hashes require a new generation output path.

Only the coordinator publishes immutable `shards/` entries. Completed shards
form a deterministic prefix, and resume rebuilds the coverage index from their
checksum-verified geometry. Ready pending proposals can be reused; incomplete
ones rerun. The distance band, duplicate policy, and implementation hashes are
recorded in `run.json`. Tests in `test_neighbor_dedup.py` include cross-shard and
cross-annotation repeats, reversed/shifted paths, crossings, extensions and an
interruption before the shard commit marker.

Validation for this change used the existing pytest environment:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_CACHE_DIR=/tmp/neighbor128-numba \
MPLCONFIGDIR=/tmp/neighbor128-mpl PYTHONPATH=../../.. \
/home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_neighbor_continuations.py tests/test_neighbor_bulk.py \
  tests/test_neighbor_mining.py tests/test_identity_sampling.py \
  tests/test_neighbor_bank.py tests/test_identity.py tests/test_regression.py \
  tests/test_refinement_logging.py
```

The suite passed 101 tests, with two skips and one deselection. A probe of the
first published real long paths accepted 128/128 draws with 4–128 voxel tails
and 128/128 draws with fixed 128-voxel tails, after history/patch/crop holdout
checks. A two-state CT-backed batch had 32 known rejection labels, no geometry
targets and four valid original-prefix seed anchors per state. The probe script
and results are saved as `sampler_probe.py` and `sampler_probe.json` inside the
new bank directory. These repeated draws verify sampler integration; they do
not establish final bank diversity or trained-model performance.

## Acceptance rules and limitations

- Find six-connected components of presence >= 0.8 whose fiber axes align with
  the target's local tangent within 25 degrees. Select two ridge points about
  20 voxels apart from the same component. Reject components touching the
  target's 2.5-voxel exclusion tube **before** modifying any masks.
- Trace between the controls using VC3D's bidirectional meeting check, then
  extrapolate 10 voxels from each end. Because only fiber directions are given,
  disable sheet-normal smoothness terms explicitly and retain the native
  isotropic smoothness term. Fiber direction channels are not sheet normals.
- Require every voxel traversed by the continuous polyline to belong to the
  same direction-filtered presence component (threshold 0.65). Check each
  visited voxel's axis against the segment tangent and densely sample
  interpolated presence. A path cannot jump a short unsupported voxel between
  regularly spaced validation samples.
- Calculate distance to the **entire continuous target annotation**, including
  segment interiors. Apply a conservative sampling bound to require >2.5 voxels
  of separation everywhere, with distance <=12 voxels. Require local target
  tangent agreement, forward progress, crop containment, and margin from the
  ends of the annotated span. Exclude the configured held-out z band from
  candidate crop selection and use training annotations only.

These checks intentionally abstain often. A connected component and matching
direction do not prove fiber identity: prediction artifacts, merged parallel
fibers, and incomplete annotations can still fool them. The guarantee is
geometric separation from the supplied annotation, not a guaranteed true
biological identity. Review these local paths before using them as negative
supervision, and retain their association with the target and local span.
Do not turn the whole component or all unannotated voxels into negatives.

The initial run attempted 69 traces and retained 12 candidates. On the existing
RelWithDebInfo build with one native tracing thread, measured native trace
latency was mean 54.9 ms, median 54.0 ms, p95 59.3 ms. The complete run, including
mining, validation and rendering, took 17.34 seconds. These are first-run
measurements with an OS page cache already used during probing, not a speedup
comparison or a cold-storage benchmark.

## Tests

The binding tests live in `volume-cartographer/python/tests/test_fiber_trace.py`;
set `VC_BUILD_PYTHON` to the build's `python` directory to test without installing.
Use a Python environment with pytest and a compatible `vc.volume` binary: loading
an older environment's `vc_core` first can cause a native ABI mismatch.

The miner tests are `tests/test_neighbor_mining.py`. They cover sparse annotation
segment interiors, continuous separation margins, unsupported voxel excursions,
incorrect fiber directions, annotation boundaries, uncut seed components, and
coordinate conversion with multi-point initialization and two-ended extrapolation.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=../../.. \
VC_BUILD_PYTHON=/home/sean/Documents/villa4/volume-cartographer/build-dev/python \
python -m pytest -q -p no:cacheprovider tests/test_neighbor_mining.py \
  /home/sean/Documents/villa4/volume-cartographer/python/tests/test_fiber_trace.py
```

Rebuild the native tests before running them if the tracer ABI has changed:

```bash
ninja -C /home/sean/Documents/villa4/volume-cartographer/build-dev test_fiber_trace3d
/home/sean/Documents/villa4/volume-cartographer/build-dev/bin/test_fiber_trace3d
```

## Live training integration

A bank path is required for identity training, including when resuming
an existing identity checkpoint:

```bash
--negative-bank output/neighbor_negatives_bulk_r12_32_l80_160_v1 \
--identity-query-patches --negative-lateral-max 32 \
--bank-following-probability .1 \
--bank-wrong-continuation-tail 4 128 \
--negative-bank-refresh-seconds 30 \
--negative-bank-cache-mb 64
```

The bank can still be generating: `complete: false` is supported. The directory
must contain the producer's `run.json`; a missing initial `bank.json` is treated
as empty coverage until publication. The bulk producer writes each path shard
completely and atomically publishes `bank.json` as chunks finish. Each data-loader
worker checks this index every 30 seconds, loads nearby paths on demand, and
keeps a bounded cache. Persistent workers discover new shards without restarting;
empty lookups are not permanently cached. Data-loader prefetch can delay use of
new entries by a few batches beyond the refresh interval.

Bank mode uses only validated, training-eligible paths. Where no usable path is
available, negative masks remain zero; the sampler does not fall back to
unverified crop components or an off-track head. Evaluation-only paths are
excluded. Annotation hashes, geometry, grid scale, CT/prediction sources, and
the held-out band must match training. Replacing a published shard or switching
the bank to another run raises an error; appending new shards is allowed.

Positive and negative queries lie directly on their respective polylines. Both
use quarter-voxel arclength interpolation, uniform selection, the same presence
threshold and full appearance support. New runs read identical CT patches for
both query classes, using the same head frame, augmentation and encoder.
There is no tube expansion or query snapping. Confidence masks mark only cells
containing line samples and require the entire cell to clear the **whole**
original target annotation. Rotation and reverse traversal need no retracing.

Synthetic wrong continuations retain an annotated prefix, use a quintic blend
with zero first/second blend derivatives at both joins, and finish
with a uniformly sampled 4–128 voxels on a verified neighboring path for new
training runs. Configure this with `--bank-wrong-continuation-tail MIN MAX`;
`128 128` fixes the tail at 128. Short paths that cannot fit the drawn tail are
rejected, not silently shortened. Bridges span 16–24 voxels for nearby paths;
when separation exceeds 12, their length is at least twice the maximum
separation. Outer-bank paths therefore need more room before the tail, and
not every 80–160 path can provide a full 128-voxel tail.
They are already-departed states:
positive geometry masks are zero and confidence teaches rejection.
Seed-segment anchors are drawn strictly from the original prefix before the
synthetic bridge. A long tail can fill all 128 recent observations with the
wrong fiber. New runs require original-fiber evidence: when fewer than two
recent patches remain, seed anchors are forced from the original prefix. If
these cannot fit, the synthetic state falls back to the original replay draw.
Length-aware draws skip shards with no fitting training
paths, and still check the actual bridge and tail after selecting geometry.
The bridge itself is never treated as a positive future. Existing crop, history,
patch and label holdout checks still apply before volume reads.

`--bank-wrong-continuation-probability .75` prefers these for 75% of recent DAgger
confirmed-departure slots. This does not change the 50/25/25 fresh/fixed/recent
allocation or any recoverable-drift/fixed stratum. A rejected or unavailable bank
proposal falls back to the original DAgger draw. Accepted synthetic states have
source 3, reported as `bank_wrong_continuation_fraction` separately from recent
replay. Candidate generation reads cached geometry only and performs no native
tracing or additional prediction-volume reads.

`--bank-following-probability .1` reserves an independent 10% of endpoint
proposals for following this bank's paths, with the normal stream loss budget.
It shares the top-level budget with `--decision-fraction`; their sum cannot exceed
one. Remaining proposals use the annotation-fresh/recent-replay mix. Missing or
unsafe bank draws fall back to that mix. Fresh targets remain actual annotations.
Their cut endpoints are unknown. Geometry and recent history
use the mined path; the original annotated parent supplies certified negatives.
All normal holdout checks apply. Source 4 is logged as `bank_following_fraction`.
Sampling revision 5 applies this endpoint-budget meaning on resumed runs too;
older revisions interpreted this option as a fraction of fresh attempts.

The tail range is saved in config/checkpoints; changing it requires a
new run. The original short bank
can still be used with `--bank-wrong-continuation-tail 4 12`. Structured training
logs report the realized `bank_wrong_continuation_tail_mean`, `_min`, and `_max`
for bank states in each logged batch; batches without bank states report null.

Checkpoints record the immutable mining-run identity and the published shard
checksums seen at the checkpoint boundary. Resume allows additional shards while
requiring the recorded shards to remain unchanged. An explicit `--negative-bank`
can attach this source to a pre-bank checkpoint; subsequent checkpoints retain
the provenance. Logs identify the bank format in `negative_source` and report
`negative_bank_shards_min/max` across the loader batches, making worker refresh
visible. Console startup output shows only the bank path, shard count and run ID;
full provenance remains in structured logs and checkpoints. Live arrivals make the exact sample sequence dependent on publication
timing; this is not a frozen-bank reproducibility mode.

The identity run was resumed from `ckpt_020000.pt` with the bank enabled;
newly generated paths need no restart. The miner remains at two workers, CPU
nice 19 and idle I/O priority. The exact launch command and pre-launch process
checks are in `output/direct_identity_run1/resume_neighbor_bank_20k.sh` and
`neighbor_resume_checks.json` in the same directory. Before launch there were
no training processes, and the generator had live workers and advancing output.

`tests/test_neighbor_bank.py` covers late publication after empty lookups,
refresh intervals, immutable updates, checkpoint growth, holdout filtering,
annotation/source validation, exact centerline sampling under rotation/reversal,
and discovery by the same persistent spawned worker. Run with the existing
ABI-compatible Python environment and pytest:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=../../.. CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nice -n 19 python -m pytest -q -p no:cacheprovider \
  tests/test_neighbor_bank.py tests/test_neighbor_continuations.py \
  tests/test_identity.py tests/test_identity_sampling.py tests/test_regression.py
```

The centerline audit used eight real locations with fine CT level 0, coarse CT
level 1 and presence level 3. All produced four positives and 32 negatives;
maximum displacement from either sampled centerline was below 1e-6 trace voxels
(float32 conversion).
See `output/neighbor_centerline_training_probe.json`. A second probe exercised
forward and reverse smooth departures, retaining 28 on-target history patches,
zero geometry targets and four known confidence rejections; its CPU optimizer
update passed (`output/neighbor_wrong_continuation_probe.json`).

The older `neighbor_negative_training_probe.json` timings describe the retired tube
sampler and are not performance measurements of the centerline implementation.
