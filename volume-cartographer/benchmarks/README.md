# Workflow benchmarks

C++ harnesses that run VC3D's real rendering paths on real data. By default they
run this branch's vc_reader pipeline on CPU and CUDA. The same sources also build
against origin/main for an optional comparison:

| pipeline        | tree                                   | sampling                                          |
|-----------------|----------------------------------------|---------------------------------------------------|
| `vc_reader-cpu` | this branch                            | `OpenedVolumes` + `vcr::grid`, CPU                |
| `vc_reader-cuda`| this branch                            | `OpenedVolumes` + `vcr::grid`, CUDA readers       |
| `chunkcache`    | origin/main (+ `bench-main-compat`), opt-in | `vc::render::ChunkCache` + `ChunkedPlaneSampler` |

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

`worker_ms` is the per-frame render worker time (`renderFrameCompleted`), and
`frames` the frames presented for the step. Each scenario runs `passes` times:
`first` meets the volume with empty decoded caches, `repeat` with the data the
first pass loaded.

Settings come from an isolated `VC3D_CONFIG_DIR` (never the user's `VC3D.ini`).

## Sources

| file | role |
|------|------|
| `bench_fixture.{hpp,cpp}`, `bench_report.cpp` | The fixture every harness goes through (no Qt). It handles arguments, the manifest, case selection, parameters, the cache and its index, cache states, loading segments and fibers, and statistics and the report. |
| `bench_session.{hpp,cpp}` | Built on the fixture, for the two viewer harnesses: the headless VC3D session (`CState`, `ViewerManager`, the volume) and settling viewers. |
| `bench_pipeline_vcr.cpp` / `bench_pipeline_chunkcache.cpp` | The tree's render pipeline (one of them is built, chosen by CMake). |

`bench_render_tifxyz` uses only the fixture, so it doesn't link VC3D.

## The manifest: `benchmarks.json`

The manifest describes the data and the cases. Relative paths are relative to the
manifest. Pass another one with `--config` (or `CONFIG=` for the scripts).

```jsonc
{
  "cache": "cache",          // where data is cached; default: cache/ beside the manifest
  "csv": "results.csv",      // optional; the CSV every timed run appends to (see Run)
  "baseline": "<label>",     // optional; the saved results compare sets the others against
  "results": [ { "label": "<label>", "path": "saved/results.jsonl" } ],  // saved results, see below
  "scratch": "...",          // optional; scratch caches for cold/partial (default: <cache>-scratch)
  "volumes":  { "<name>": { "url": "s3://..." } },
  "segments": { "<name>": { "url": "s3://.../x.tifxyz", "volume": "<volume>", "scale": 1 } },
  "fibers":   { "<name>": { "path": "/local/folder", "volume": "<volume>", "scale": 1 } },
  "defaults": { "cache_gb": 16, "passes": 2, "cache_state": "cached", ... },
  "benchmarks": {
    "viewer_navigation": [ { "name": "<case>", "segment": "<segment>", "width": 960, ... } ],
    "line_annotation":   [ { "name": "<case>", "fibers": "<fibers>", "count": 4, ... } ],
    "render_tifxyz":     [ { "name": "<case>", "segment": "<segment>", "crop": [x, y, w, h], ... } ]
  }
}
```

- Each **data item** has either a `url` (remote; `prefetch.sh` copies it into
  the cache) or a `path` (local; read in place). Segments and fibers name the
  `volume` their coordinates belong to, and give the `scale` that maps them to
  that volume's level-0 voxels. A case's volume is its own `volume` key, or else
  its segment's or fibers' volume.
- Each benchmark has a list of **cases**. A harness runs one case per process:
  use `--case <name>` (optional when there is only one case), and `--list-cases`
  to print the names.
- **Parameters** are resolved in this order: `--kebab-case` on the command line,
  then the case's `snake_case` key, then `defaults`, then the harness's built-in
  value. A case key or option that nothing reads is an error, which catches typos.

| parameter | used by | default |
|-----------|---------|---------|
| `backend` (`cpu`/`cuda`) | all | `cpu` |
| `cache_state` (`cached`/`partial`/`cold`) | all | `cached` |
| `partial_levels` | all, state `partial` | every level but 0 |
| `cache_gb`, `gpu_cache_gb` | all | 16, 8 |
| `passes`, `timeout_ms`, `max_resolution` | viewer harnesses | 2, 120000, 0 |
| `fetch_max_gb` | viewer harnesses, remote states | 100 |
| `prefetch_margin` | viewer harnesses, `--prefetch` | 1.5 |
| `width`, `height`, `view_scale`, `pan_steps`, `zoom_steps`, `slice_steps`, `jumps`, `burst` | `viewer_navigation` | 960, 540, 1, 8, 4, 8, 8, 60 |
| `count`, `steps`, `strip_width`, `strip_height`, `cut_size` | `line_annotation` | 4, 24, 1600, 320, 640 |
| `group`, `scale`, `slices`, `crop`, `surface_interpolation`, `iterations`, `tool`, `work_dir` | `render_tifxyz` | 2, 1, 21, none, linear, 3, this tree's tool, `$TMPDIR/vc_bench_render_tifxyz` |

Options that aren't parameters: `--config`, `--case`, `--list-cases`, `--json <file>`
and `--csv <file>` (append results), `--run-id`, `--commit`, `--dump <dir>` (save viewer frames as PNG), `--prefetch`,
`--keep-scratch`.

## Cache states

`cache_state` sets what is on disk when a run starts:

| state | volume source | needs |
|-------|---------------|-------|
| `cached`  | The volume's URL, read from the cache offline (VC3D's `perf/remote_offline`): nothing is fetched. A run that reads a chunk the cache lacks fails (exit 3) and records nothing; `vc_render_tifxyz` fails on it. origin/main's tree opens the cached copy as a local folder and cannot check. | `prefetch.sh` |
| `partial` | The volume's URL, through a scratch cache seeded with the cached copy's metadata and `partial_levels` (hard links). The rest is fetched during the run. | `prefetch.sh`, network |
| `cold`    | The volume's URL, through an empty scratch cache. Everything is fetched during the run. | network |

Scratch caches are removed when the run ends (`--keep-scratch` keeps them). Only
the volume's state varies: segments and fibers are inputs loaded before timing,
and always come from the cache. Within one run the scratch cache persists, so
`repeat` passes and `bench_render_tifxyz`'s repeat runs read what the first one
fetched. `fetched_chunks` and `fetched_mb` in the report show the traffic.
origin/main's tree only supports `cached`. Seeding `partial` with every level
but 0 leaves a `render_tifxyz` case at `group` 2 fully on disk, so for that case
pick `partial_levels` without its group.

## Prefetch

```sh
volume-cartographer/benchmarks/prefetch.sh                    # every case of every harness
volume-cartographer/benchmarks/prefetch.sh viewer_navigation  # one harness
```

`prefetch.sh` needs a vc_reader build. It never fetches what the cache already
holds, so after a parameter change it fetches only the difference, and on a
complete cache it only confirms that, in seconds. It does the following:

1. Downloads the cases' remote segments and fiber folders into the cache
   (`vcr::cache::downloadPrefix`).
2. **Plans** each viewer case instead of replaying it online. A plan pass drives
   the case's inputs against the cache offline in `fill` mode, where a chunk the
   cache lacks reads as empty at once and is recorded
   (`vcr::Array::offline_misses`). So no step waits on the network, and every
   missing chunk shows up. The pass then fetches exactly the recorded chunks (`vcr::cache::prefill` with a
   chunk list) and exits 2. Passes repeat until one finds nothing missing (exit
   0), at most `MAX_PASSES` (6). Each case is planned exactly as its timed runs
   read (margin 1, the case's passes) on each backend in `BACKENDS` (`cpu cuda`;
   CUDA sampling reads chunks the CPU does not), then with grown views (one
   pass). The grown views don't replace the exact plan, because a
   larger viewer pans by a larger step and so visits other places.
3. Runs each `render_tifxyz` case once on the URL through the cache. The tool's
   reads are what its timed runs read.

Each item's cached location (`vcr::zarr::decodedCachePath` of its URL) is recorded
in `<cache>/benchmark-index.json`. Every tree, origin/main included, finds the
cached copies through that index.

## Build

```sh
# this branch (vc_reader, CUDA)
cmake -S volume-cartographer -B volume-cartographer/build -DVC_BUILD_BENCHMARKS=ON
ninja -C volume-cartographer/build bench_viewer_navigation bench_line_annotation bench_render_tifxyz
```

## Run

```sh
volume-cartographer/benchmarks/run.sh                                  # cpu, cuda, compare (with saved results)
STATES="cached partial cold" volume-cartographer/benchmarks/run.sh     # all three cache states
EXTRA="--passes 1" volume-cartographer/benchmarks/run.sh cpu compare   # one pipeline again, then the table
BENCHES=line_annotation volume-cartographer/benchmarks/run.sh cuda     # one harness
```

`run.sh` runs every case of the manifest for each harness, step and cache state.
Every run appends one row per step and metric to two files:

- **the CSV** (see below), for spreadsheets and quick comparisons across runs. Columns:
  `timestamp, run_id, commit, benchmark, case, pipeline, cache_state, step,
  metric, n, mean, p50, p95, min, max, total, config`. The header is written
  when the file is new, and `config` is the run's configuration as one JSON
  string.
- **`$OUT/results.jsonl`** (default `benchmarks/out/`), the same rows as JSON
  lines, which `compare` reads.

`timestamp` is when the harness started (UTC). `run_id` is the same for every row
of one `run.sh` invocation (`RUN_ID`, default a UTC timestamp), so the rows of
one run can be filtered together. `commit` is the short hash of the build tree's
source, with `-dirty` when tracked files had uncommitted changes. `config` holds
the fixture facts and every parameter's effective value. Each harness also prints
a table.

Which CSV a run appends to, first match wins:

1. `--csv <file>`.
2. The manifest's `csv` (`benchmarks/results.csv` here). Every timed run of any
   harness appends to it, whether from `run.sh` or run by hand, so it collects
   the whole history in one file. `--prefetch` replays don't write to it.
3. The `--json` file's path with `.csv`.

`--run-id` and `--commit` set those columns (default: the timestamp and
`unknown`).

## Saved results and baselines

`compare` (`compare.py --config <manifest> [results.jsonl]`) shows this run's
means beside the means of any earlier results listed in the manifest, so you
don't have to rerun a baseline:

```json
"baseline": "main-1101",
"results": [
  { "label": "main-1101", "path": "saved/main-1101.jsonl" },
  { "label": "before-fix", "path": "/mnt/raid_nvme/dev_builds/bench_out/results.jsonl" }
]
```

- **Saving a result:** any `results.jsonl` written by `run.sh` (or `--json`)
  works. Copy it somewhere, or point at it where it is. Paths are relative to
  the manifest.
- **The baseline:** the entry `baseline` names, else the first entry. Every
  other column, this run's included, is followed by `d% <column>`, its change
  against the baseline in percent (negative means lower than the baseline).
- **Matching pipelines:** a column is compared with the baseline's same
  pipeline. If the baseline holds only one pipeline (a saved origin/main
  `chunkcache` run, say), every column is compared with that one, so a saved
  main run is a baseline for both CPU and CUDA.
- **Rows:** a row is (benchmark, case, cache state, step, metric). Rows that only
  some sources have show `-` for the others. In each file, the latest line of a
  (row, pipeline) pair wins.
- **Configuration notes:** under the table, `compare` lists configuration
  differences against the baseline for the same benchmark, case and cache state:
  parameters, data and settings, but not machine paths or traffic counters. A
  difference there means the two columns measured different workloads.

```sh
volume-cartographer/benchmarks/run.sh main             # once: produce a main result
cp volume-cartographer/benchmarks/out/results.jsonl volume-cartographer/benchmarks/saved/main-1101.jsonl
# list it under "results" (and as "baseline"), then from now on:
volume-cartographer/benchmarks/run.sh                  # cpu, cuda, compare against main-1101
```

## Comparing with origin/main (optional)

```sh
# a worktree with the compat commit (VC3D object library, renderSettled(), this folder),
# e.g. branch bench-main-compat
git worktree add /mnt/raid_nvme/dev_builds/villa-main bench-main-compat
rsync -a --delete --exclude cache/ --exclude cache-scratch/ --exclude out/ \
      volume-cartographer/benchmarks/ /mnt/raid_nvme/dev_builds/villa-main/volume-cartographer/benchmarks/
cmake -G Ninja -S /mnt/raid_nvme/dev_builds/villa-main/volume-cartographer \
      -B /mnt/raid_nvme/dev_builds/vc-main -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_PYTHON=OFF \
      -DVC_BUILD_BENCHMARKS=ON
ninja -C /mnt/raid_nvme/dev_builds/vc-main bench_viewer_navigation bench_line_annotation bench_render_tifxyz

volume-cartographer/benchmarks/run.sh cpu cuda main compare   # MAIN_BUILD selects the tree
```

Repeat the rsync after changing this folder. The `main` step passes this
folder's manifest, so both trees read the same cached copies through the index.
It runs cache state `cached` only. origin/main must never be pointed at the cache
as its *remote* cache: its persistent-cache setup deletes a folder without its
own format marker. Here it only opens the cached copy as a local folder.

## Limits

- The line annotation harness builds the views the controller builds but not
  the controller itself (no object store, overlays, or solve): the cut frame
  uses the line's display-frame normal as the dialog does without a manual
  rotation.
- Page-cache state of the local volume files is not controlled: `first` passes
  start with empty decoded caches, but the disk is not cold.
- `cold` and `partial` include real network time, which varies from run to run.
  Compare them over several runs.
- Plan passes find the chunks the viewers request with views grown by
  `prefetch_margin`. origin/main's pipeline can choose other levels than that
  covers, and its tree cannot detect a miss, so its `cached` runs rely on the
  margin.
- `cached` runs on vc_reader read the cache as VC3D reads a remote volume
  (the decoded cache of its URL). Before this, they read the cached copy as a
  local folder, so compare such results only with runs that read the same way.
- The branch links vc_reader as `libvc_reader_native.so` (`VC_READER_SHARED_READER`).
