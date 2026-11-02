# Workflow benchmarks

C++ harnesses that run VC3D's real rendering paths on real data, built from the
same sources against two pipelines:

| pipeline        | tree                                   | sampling                                          |
|-----------------|----------------------------------------|---------------------------------------------------|
| `chunkcache`    | origin/main (+ `bench-main-compat`)    | `vc::render::ChunkCache` + `ChunkedPlaneSampler`  |
| `vc_reader-cpu` | this branch                            | `OpenedVolumes` + `vcr::grid`, CPU                |
| `vc_reader-cuda`| this branch                            | `OpenedVolumes` + `vcr::grid`, CUDA readers       |

| harness                   | workflow                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| `bench_viewer_navigation` | VC3D's layout: XY/XZ/YZ slice viewers + the flattened segmentation viewer. Open, then per viewer arrow-key pans, zoom steps, slice scrolling; focus jumps (all four viewers); a held-key pan burst at 60 Hz input |
| `bench_line_annotation`   | The Line Annotation window on the longest fibers: `buildLineViewSurfaces` (strip, side strip) and the current/side cut planes in four annotation viewers, opened at the fiber middle, then stepped along the whole fiber |
| `bench_render_tifxyz`     | This tree's `vc_render_tifxyz` end to end (subprocess) on a segment crop |

The viewer harnesses create real `CChunkedVolumeViewer`s through `ViewerManager`
on the offscreen Qt platform, linked from VC3D's own objects (the `vc3d_app`
object library), and drive them through their public input methods. Each step is
timed from the input to:

- `first_frame_ms` — the first frame any affected viewer presents;
- `settled_ms` — every affected viewer shows its final frame and no volume read
  is outstanding (`CChunkedVolumeViewer::renderSettled()`, true twice with the
  event queue drained in between).

`worker_ms` is the per-frame render worker time (`renderFrameCompleted`),
`frames` the frames presented for the step. Each scenario runs `--passes` times:
`first` meets the volume with empty decoded caches, `repeat` with the data the
first pass loaded.

Settings come from an isolated `VC3D_CONFIG_DIR` (never the user's `VC3D.ini`);
`--max-resolution` sets the viewer's max displayed resolution (default 0).

## Data and prefetching

Timed runs open the cached Open Data volume folder as a **local** OME-Zarr
(`.../volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-...zarr`),
so neither pipeline fetches anything and both read the same files. origin/main
must never be pointed at that cache as its *remote* cache: its persistent-cache
setup deletes a folder without its own format marker.

The cache only holds what has been read before, so first run the scenarios on
the vc_reader build with `--prefetch`: the volume is opened as its `s3://`
source with the cache's root as VC3D's cache root, and the scenario replays with
views grown by `--prefetch-margin` (1.5), and stops past `--prefetch-max-gb` (100) fetched, to cover what either pipeline's level
and fallback choices read around them. A second prefetch round must report
`fetched_chunks: 0`. `bench_render_tifxyz --prefetch` does the same for the
tool. On main, `known_empty_chunks` counts chunks a run found absent (genuinely
empty chunks of the masked volume count too).

Fibers are VC3D fiber JSON (`line_points` in level-0 voxels of the same volume),
from `/mnt/raid_nvme/spiral_dataset_working/fibers`. The segment mesh is a
level-2 tifxyz; `--segment-scale 4` maps it to level-0 voxels.

## Build

```sh
# this branch (vc_reader, CUDA)
cmake -S volume-cartographer -B volume-cartographer/build -DVC_BUILD_BENCHMARKS=ON
ninja -C volume-cartographer/build bench_viewer_navigation bench_line_annotation bench_render_tifxyz

# origin/main: a worktree with the compat commit (VC3D object library,
# renderSettled(), this folder), e.g. branch bench-main-compat
git worktree add /mnt/raid_nvme/dev_builds/villa-main bench-main-compat
cmake -G Ninja -S /mnt/raid_nvme/dev_builds/villa-main/volume-cartographer \
      -B /mnt/raid_nvme/dev_builds/vc-main -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_PYTHON=OFF \
      -DVC_BUILD_BENCHMARKS=ON
ninja -C /mnt/raid_nvme/dev_builds/vc-main bench_viewer_navigation bench_line_annotation bench_render_tifxyz
```

After changing this folder, copy it to the main worktree
(`rsync -a --delete benchmarks/ /mnt/raid_nvme/dev_builds/villa-main/volume-cartographer/benchmarks/`).

## Run

```sh
volume-cartographer/benchmarks/run_ab.sh              # prefetch, main, cpu, cuda, compare
volume-cartographer/benchmarks/run_ab.sh cpu compare  # one pipeline again, then the table
```

Results append to `$OUT/results.jsonl` (one line per step and metric: n, mean,
p50, p95, min, max, and the run's configuration); each harness also prints a
table. `compare` lays out the means of every pipeline side by side.

Common options: `--volume`, `--backend cpu|cuda`, `--cache-gb` (decoded RAM
cache, 16), `--gpu-cache-gb` (8), `--passes` (2), `--timeout-ms`, `--json`.
`bench_viewer_navigation`: `--segment`, `--segment-scale`, `--width/--height`
(960x540 per viewer), `--pan-steps` (8), `--zoom-steps` (4), `--slice-steps` (8),
`--jumps` (8), `--burst` (60). `bench_line_annotation`: `--fibers`,
`--fiber-count` (4), `--steps` (24), `--strip-width/--strip-height` (1600x320),
`--cut-size` (640). `bench_render_tifxyz`: `--segment`, `--group`, `--scale`,
`--slices`, `--crop x,y,w,h`, `--surface-interpolation` (linear; the trees'
defaults differ), `--iterations` (3), `--work-dir`.

## Limits

- The line annotation harness builds the views the controller builds but not
  the controller itself (no object store, overlays, or solve): the cut frame
  uses the line's display-frame normal as the dialog does without a manual
  rotation.
- Page-cache state of the local volume files is not controlled; `first` passes
  are "decoded caches empty", not "disk cold".
- The branch links vc_reader as `libvc_reader_native.so` (`VC_READER_SHARED_READER`).
