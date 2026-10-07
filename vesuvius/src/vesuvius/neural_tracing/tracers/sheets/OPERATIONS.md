# Sheet tracing: operations guide

This guide covers where the models are, how to run them and how to train on the H100. The rest of the documentation
is elsewhere:

* the model itself: [ARCHITECTURE.md](ARCHITECTURE.md);
* the options: [README.md](README.md);
* every experiment with its numbers: [output/remote/RESULTS.md](output/remote/RESULTS.md).

Paths below are relative to this package (`vesuvius/src/vesuvius/neural_tracing/tracers/sheets/`) unless absolute.
State as of 2026-10-07.

## 1. Models

Two models are supported. Both share one architecture (18.5 M parameters) and read only the CT, at level 0.

| model | checkpoint | what it is |
| --- | --- | --- |
| **`remote_l0_points_ft`** | `output/remote/remote_l0_points_ft/last.pt` | the sheet tracer, fine-tuned on 5 to 16-point fronts |
| **`remote_l0_fiber_ft`** | `output/remote_l0_fiber_ft/` on the H100 | `points_ft` plus fiber-seeded episodes, fiber tokens and the fiber-following mode (3 to 9 points) |

`points_ft` results (H100, 60 steps, threshold 0.4, held-out scoring; RESULTS.md, "Front points"), as coverage /
precision / traces with a real switch:

| set | 5 points | 9 points | 15 points |
| --- | --- | --- | --- |
| Scroll 1 | 0.848 / 0.968 / 10 | 0.824 / 0.984 / 15 | 0.783 / 0.940 / 33 |
| PHerc0139 slab | 0.457 / 0.980 / 23 | 0.605 / 0.980 / 39 | 0.548 / 0.957 / 48 |
| PHerc1667 slab | 0.561 / 0.980 / 35 | 0.580 / 0.972 / 47 | 0.579 / 0.957 / 55 |

Checkpoints of the same architecture trained along the way (`l0_joint`, `remote_l0_joint_scrolls`,
`remote_l0_mix_scratch`) still load. Checkpoints of the removed model types (m7, context, profile, smoothing and
others) do not: their configs are rejected.

## 2. Where the checkpoints are

| location | path | contents |
| --- | --- | --- |
| this machine | `output/remote/<run>/` | copies of H100 runs: `last.pt`, kept `ckpt_NNNNNN.pt`, `log.jsonl`, `config.json`, `training.png`, validation figures |
| H100 | `/ephemeral/sean/sheet_tracing/villa/vesuvius/src/vesuvius/neural_tracing/sheet_tracing/output/<run>/` | the originals, including `best.pt` |

A checkpoint is a dict with these keys:

* `model`: the raw weights;
* `ema`: the EMA weights;
* `opt`, `config`, `iteration`, `minutes`, `best`.

`evaluate.load_checkpoint` builds the model from the stored config and loads the EMA weights. Benchmarks use
`last.pt`.

**Paths inside H100 checkpoints.** Their configs record the H100's data paths (`/ephemeral/sean/sheet_tracing/data/...`).
On this machine, pass `--path-map OLD=NEW` (to `infer.py`, `trace_benchmark.py`, `hard_spots.py`
and `scripts/trace_from_fibers.py`), or use `scripts/bench_local.sh`, which has the mappings built in:

| H100 path | this machine |
| --- | --- |
| `/ephemeral/sean/sheet_tracing/data/s1_ds2.zarr` | `/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr` |
| `/ephemeral/sean/sheet_tracing/data/verified_patches_old` | `/home/sean/Desktop/spiral_dataset/verified_patches` |
| `/ephemeral/sean/sheet_tracing/data/verified_patches_working` | `/home/sean/Desktop/spiral_dataset_working/verified_patches` |

The PHerc0139 and PHerc1667 patches and volumes (`data/unseen/`) and the fiber data (`data/afv/`) exist only on the
H100.

## 3. Running a model (this machine)

Run from `vesuvius/` with `.venv/bin/python`. First set `M` to the H100 data folder:
`M=/ephemeral/sean/sheet_tracing/data`.

**Trace a surface from a seed line** with `infer.py`. It writes a tifxyz with one row per step:

```bash
.venv/bin/python -m vesuvius.neural_tracing.tracers.sheets.infer \
  --checkpoint src/vesuvius/neural_tracing/tracers/sheets/output/remote/remote_l0_points_ft/last.pt \
  --seed-patch /path/to/a/tifxyz --line row:40 --range 100:160 --direction + --steps 60 --threshold 0.5 \
  --out /path/to/output_tifxyz \
  --path-map $M/s1_ds2.zarr=/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr
```

* **Seed:** either a grid line of an existing tifxyz or a JSON file.
  * Grid line: `--line row:N` or `col:N`, with columns `--range lo:hi`, advancing `--direction +` or `-` (towards
    increasing or decreasing rows).
  * JSON: `--seed-json` with `{"points": [[x, y, z], ...], "direction": [dx, dy, dz]}` in level-0 voxels.
* **Direction:** needed to orient the first crops; it should lie in the sheet.
* **Stopping:** tracing stops at the first row with no confident point connected to the surface.
* **Front size and threshold:** the front size follows from the seed length (60 vx per point).
  `scripts/pick_inference.py` picked 15 points at threshold 0.5 for the PHercParis4 runs.

**Trace from fibers** with `scripts/trace_from_fibers.py`, from VC3D fiber JSONs:

* `--points 15 --direction both` traces sheets up and down from horizontal fibers.
* With `remote_l0_fiber_ft`:
  * `--fiber-pool POOL` shows the model the fibers around each trace;
  * `--follow` runs the fiber-following mode and writes each extended fiber as a VC3D fiber JSON to `OUT/fibers/`.

**Benchmark models against each other** with `scripts/bench_local.sh` (100 held-out seeds, threshold 0.4, all
models in one run):

```bash
src/vesuvius/neural_tracing/tracers/sheets/scripts/bench_local.sh \
  src/vesuvius/neural_tracing/tracers/sheets/output/remote/trace_bench/my_test_s60_t0.4 60 \
  pts=remote_l0_points_ft/last.pt fiber=remote_l0_fiber_ft/last.pt
```

* **Output:** `report.json`, `per_step.png`, per-seed figures, and `tifxyz/` (`seedNN_gt` and `seedNN_<model>`) for
  VC3D.
* **Comparing:** compare only models benchmarked on the same machine. The GB10 and the H100 differ slightly in bf16,
  which flips a few threshold decisions over a trace, so local and H100 numbers must not be mixed. Each machine is
  deterministic run to run.
* **Re-scoring:** `trace_benchmark.py --rescore DIR` re-scores the switch counts of existing folders from their
  tifxyz.

**Mine failure spots:** `hard_spots.py --checkpoint ... --seeds 1500 --out FILE.json --path-map ...`. It writes
progress every 100 traces, so it can be stopped early.

## 4. Training

```bash
python -m vesuvius.neural_tracing.tracers.sheets.train --config CONFIG.json --name RUN [--init CKPT] [--set section.field=value ...]
python -m vesuvius.neural_tracing.tracers.sheets.train --resume output/RUN/last.pt
```

* **`--init`:** starts from a checkpoint's EMA weights with a fresh optimizer and schedule. The fiber and mode
  modules start fresh when the checkpoint lacks them; the mode embedding starts at zero, so the model begins
  unchanged.
* **`--resume`:** continues a run, including the optimizer and the elapsed time.
* **Configs:** `configs/remote_l0_points_ft.json` and `configs/remote_l0_fiber_ft.json` (H100 paths).
* **Option reference:** README.md.

The patch index is cached in `output/cache/`. It is rebuilt automatically when the patch directories or
`data/patches.py`'s `INDEX_VERSION` change, and it measures each patch's real cell size where `meta.json` is wrong.

## 5. The H100 machine

* **Login:** `ssh ubuntu@147.224.219.64`. It's a shared machine, so other people's jobs run on it.
* **GPU: use GPU 7 only.** `remote_l0_fiber_ft` (about 160 points per batch, 32 workers) takes about 31 GB and runs
  at about 1.4 s per iteration.
* **Everything stays under `/ephemeral/sean/sheet_tracing`**, including temporary files and caches. The root disk is
  shared and nearly full: do not write anywhere else (plain logs and text are tolerated, nothing else).
* **Always `source /ephemeral/sean/sheet_tracing/env.sh` first.** It does the following:
  * points every temp and cache directory (tmp, uv, pip, matplotlib, torch, triton, inductor, numba, CUDA) into
    `cache/`;
  * sets `PYTHONPATH` to the repo copy and `$PY` to the venv;
  * sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* **Other people's resources:** watch the shared memory that DataLoader workers use (each holds `train.prefetch`
  batches of blocks). Large batches with many workers can break other people's runs.

**Layout** of `/ephemeral/sean/sheet_tracing`:

| path | what |
| --- | --- |
| `villa/vesuvius/src/vesuvius/neural_tracing/sheet_tracing/` | the code (a copy of this package); `output/` holds the runs, `configs/` the configs |
| `data/` | `s1_ds2.zarr` (levels 0 and 1 decompressed in place), both Scroll 1 patch sets, `cut_windings_square/`, `unseen/PHerc0139` and `unseen/PHerc1667` (volumes and train patches), `afv/` (the PHercParis4 AFV, `paris4_H_fiber_runs.jsonl`, `paris4_fiber_pool/`), `fibers/`, `umbilicus_s1_ds2.json` |
| `logs/` | `RUN.log` (training output, tqdm with `\r`) and `RUN.pid` |
| `env.sh`, `launch.sh` | the environment, and the detached launcher |
| `cache/` | all temp and cache files |

**Sync the code** after changing it here. Configs are excluded because they are created on the H100:

```bash
cd vesuvius/src/vesuvius/neural_tracing/sheet_tracing
rsync -a --exclude output/ --exclude __pycache__/ --exclude configs/ ./ \
  ubuntu@147.224.219.64:/ephemeral/sean/sheet_tracing/villa/vesuvius/src/vesuvius/neural_tracing/sheet_tracing/
```

**Launch** (detached; survives logout):

```bash
ssh ubuntu@147.224.219.64
source /ephemeral/sean/sheet_tracing/env.sh
O=$B/villa/vesuvius/src/vesuvius/neural_tracing/sheet_tracing/output
$B/launch.sh 7 configs/remote_l0_fiber_ft.json my_run --init $O/remote_l0_points_ft/last.pt   # GPU, config, name, train args
```

**Monitor:**

```bash
tr '\r' '\n' < $B/logs/my_run.log | grep -E "val score|Traceback|Error|time budget"   # validations, errors
tr '\r' '\n' < $B/logs/my_run.log | tail -1                                         # progress line
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

The H100's clock and logs are in UTC (Chicago is UTC-5 in summer).

**Stop:**

* Send `kill -TERM $(cat $B/logs/my_run.pid)`. The trainer exits cleanly and shuts down its loader workers.
* Do not use `pkill -f PATTERN`: it also matches any shell whose command line contains the pattern, including the
  one running it.
* Afterwards, check for leftover workers with `ps -eo pid,ppid,args | grep multiprocess | grep ephemeral/sean/villa`.

**Copy a finished run back here** (the bulky kept checkpoints are optional):

```bash
R=ubuntu@147.224.219.64:/ephemeral/sean/sheet_tracing
O=$R/villa/vesuvius/src/vesuvius/neural_tracing/sheet_tracing/output
L=vesuvius/src/vesuvius/neural_tracing/sheet_tracing/output/remote
rsync -a --exclude '*.pt' $O/my_run $L/ && rsync -a --include 'ckpt_*.pt' --include 'last.pt' --exclude '*' $O/my_run/ $L/my_run/
rsync -a $R/logs/ $L/logs/
```

Then benchmark it, and record the result in `output/remote/RESULTS.md`.
