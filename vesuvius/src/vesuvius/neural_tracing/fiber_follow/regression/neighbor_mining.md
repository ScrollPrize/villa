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

Add these arguments to the identity training command, including when resuming
an existing identity checkpoint:

```bash
--negative-bank output/neighbor_negatives_bulk_v1 \
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

For each training state, the loader builds a narrow (0.6-voxel) tube around
nearby validated paths. It checks current crop presence and requires the whole
cell used for a confidence label to clear the exclusion distance from the
**entire** original annotation. Positive and negative embedding queries retain
the existing matching interpolation phase. Shifted query coordinates receive a
second whole-annotation clearance check. Rotation and reverse traversal are
supported; neither requires retracing.

Checkpoints record the immutable mining-run identity and the published shard
checksums seen at the checkpoint boundary. Resume allows additional shards while
requiring the recorded shards to remain unchanged. An explicit `--negative-bank`
can attach this source to a pre-bank checkpoint; subsequent checkpoints retain
the provenance. Logs identify `negative_source: native_path_bank_v1` and report
`negative_bank_shards_min/max` across the loader batches, making worker refresh
visible. Live arrivals make the exact sample sequence dependent on publication
timing; this is not a frozen-bank reproducibility mode.

The currently running trainer must load this code on its next launch/resume.
Once launched with the bank enabled, newly generated negatives require no further
restart. The miner continues separately with two workers, CPU nice 19, and idle
I/O priority.

`tests/test_neighbor_bank.py` covers late publication after empty lookups,
refresh intervals, immutable updates, checkpoint growth, holdout filtering,
annotation/source validation, forward/reverse sampling, interpolation phase,
and discovery by the same persistent spawned worker. Run with the existing
ABI-compatible Python environment and pytest:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=../../.. CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nice -n 19 python -m pytest -q -p no:cacheprovider \
  tests/test_neighbor_bank.py tests/test_identity.py \
  tests/test_identity_sampling.py tests/test_regression.py
```

A real-data CPU probe used eight locations from the growing bank, fine CT level
0, coarse CT level 1, and presence level 3. All eight produced 26–32 usable
negative queries (four positive queries each); a full optimizer update had a
nonzero identity objective and completed successfully. Measurements are saved
in `output/neighbor_negative_training_probe.json`. With one CPU thread at nice
19 while the other training job was active, 24 repeated label-generation calls
averaged 33.6 ms (median 20.9, p95 103.4) for the bank and 46.3 ms (median 24.3,
p95 70.0) for the legacy sampler on the same eight crops. This small probe checks
integration and overhead, not training quality or a general throughput claim.
