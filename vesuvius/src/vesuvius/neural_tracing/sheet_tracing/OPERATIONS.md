# Sheet tracing: operations guide

Where the models are, how to run them, how to train on the H100, and which models are best. The model itself is
described in [ARCHITECTURE.md](ARCHITECTURE.md), the options in [README.md](README.md), and every experiment with its
numbers in [output/remote/RESULTS.md](output/remote/RESULTS.md). Paths below are relative to this package
(`vesuvius/src/vesuvius/neural_tracing/sheet_tracing/`) unless absolute. State as of 2026-10-05.

## 1. Best models

All current models share one architecture (the m7 model, 7.5 M parameters) and need two volumes: CT and the m7
surface prediction, both read at pyramid level 1. Use threshold **0.4** for tracing (`--threshold 0.4`). The
numbers are the 100-seed local benchmark on corrected seeds (folders `output/remote/trace_bench/m7_fix100_local_*`).

| model | checkpoint | 60 steps: coverage / real switches / off-sheet points | 30 steps: coverage / real switches / off-sheet points | use it for |
| --- | --- | --- | --- | --- |
| **`m7_hard`** | `output/remote/m7_hard/last.pt` | **0.894** / 14 / 732 | **0.803** / 20 / 1,113 | furthest reach; more mistakes on shorter traces |
| **`m7_fix`** | `output/remote/m7_fix/last.pt` | 0.850 / **10** / **164** | 0.770 / 13 / 830 | safest long traces (precision 0.999) |
| `m7_len2` | `output/remote/ctx2_m7_len2/last.pt` | 0.862 / 12 / 588 | 0.784 / **9** / 373 | balanced; the base of the two above |
| `m7` | `output/remote/ctx2_m7_len/last.pt` | 0.846 / 11 / 223 | 0.769 / **9** / **361** | the original m7 model |
| `ctx2` | `output/remote/big_curve_ctx2/last.pt` | 0.807 / 13 / 563 | 0.748 / 12 / 707 | best CT-only model (no m7 volume needed) |

In short: **`m7_hard` if reach matters most, `m7_fix` if mistakes matter most.**

* `m7_hard` = `m7_len2` fine-tuned on corrected patch scales, with a third of training episodes at 466 mined failure
  spots, plus EMA.
* `m7_fix` = the same without the failure spots.
* Both store EMA weights (the default when loading) and raw weights (`--raw NAME` in the benchmark); the two perform
  about the same.
* Tried and not better: level-0 CT, fiber presence (at level 0 and at level 1), the normal profile with long
  memory, live-chain training, capacity beyond 7.5 M, and 20-step rollouts. See RESULTS.md.

## 2. Where the checkpoints are

| location | path | contents |
| --- | --- | --- |
| this machine | `output/remote/<run>/` | copies of every H100 run: `last.pt`, kept `ckpt_NNNNNN.pt` (every 5,000 iterations), `best.pt` for older runs, `log.jsonl`, `config.json`, `training.png`, validation figures |
| H100 | `/ephemeral/sean/sheet_tracing/villa/vesuvius/src/vesuvius/neural_tracing/sheet_tracing/output/<run>/` | the originals, including `best.pt` for the recent runs (not copied) |
| this machine | `output/<run>/` | early local runs (`tiny_*`, `hermite_grid_long_v1`) |

A checkpoint is a dict with:
* `model`: raw weights;
* `ema`: EMA weights, only for runs with `train.ema`;
* `opt`, `config`, `iteration`, `minutes`, `best`.

`evaluate.load_checkpoint` builds the model from the stored config and loads the EMA weights when present. All
benchmarks used `last.pt`. `best.pt` is picked by validation score, which has been a poor guide to trace quality.

**Paths inside H100 checkpoints.** Their configs record the H100's data paths (`/ephemeral/sean/sheet_tracing/data/...`).
On this machine, pass `--path-map OLD=NEW` (to `infer.py`, `trace_benchmark.py`, `trace_spread.py`, `hard_spots.py`)
or use `scripts/bench_local.sh`, which has the mappings built in:

| H100 path | this machine |
| --- | --- |
| `/ephemeral/sean/sheet_tracing/data/s1_ds2.zarr` | `/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr` |
| `/ephemeral/sean/sheet_tracing/data/m7_ds2.zarr` | `/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/m7_ds2.zarr` |
| `/ephemeral/sean/sheet_tracing/data/fiber_presence.ome.zarr` | `/home/sean/Desktop/spiral_dataset/fiber_zarrs/PHercParis4-20260411134726-las-sd1-d74cca79_presence.ome.zarr` |
| `/ephemeral/sean/sheet_tracing/data/verified_patches_old` | `/home/sean/Desktop/spiral_dataset/verified_patches` |
| `/ephemeral/sean/sheet_tracing/data/verified_patches_working` | `/home/sean/Desktop/spiral_dataset_working/verified_patches` |

## 3. Running a model (this machine)

Run from `vesuvius/` with `.venv/bin/python`. Set `M` to the H100 data folder once:
`M=/ephemeral/sean/sheet_tracing/data`.

**Trace a surface from a seed line** (`infer.py`; writes a tifxyz, one row per step, 25 columns):

```bash
.venv/bin/python -m vesuvius.neural_tracing.sheet_tracing.infer \
  --checkpoint src/vesuvius/neural_tracing/sheet_tracing/output/remote/m7_fix/last.pt \
  --seed-patch /path/to/a/tifxyz --line row:40 --range 100:160 --direction + --steps 60 --threshold 0.4 \
  --out /path/to/output_tifxyz \
  --path-map $M/s1_ds2.zarr=/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr \
  --path-map $M/m7_ds2.zarr=/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/m7_ds2.zarr
```

* **Seed:** a grid line of an existing tifxyz (`--line row:N` or `col:N`, columns `--range lo:hi`, advance
  `--direction +` or `-` towards increasing or decreasing rows), or `--seed-json` with
  `{"points": [[x, y, z], ...], "direction": [dx, dy, dz]}` in level-0 voxels.
* **Direction:** needed to orient the first crops; it should lie in the sheet.
* **Stopping:** tracing stops at the first row with no confident point connected to the surface.

**Benchmark models against each other** (`scripts/bench_local.sh`: 100 held-out seeds, threshold 0.4, all models
in one run):

```bash
src/vesuvius/neural_tracing/sheet_tracing/scripts/bench_local.sh \
  src/vesuvius/neural_tracing/sheet_tracing/output/remote/trace_bench/my_test_s60_t0.4 60 \
  m7=ctx2_m7_len/last.pt fix=m7_fix/last.pt hard=m7_hard/last.pt -- --raw hard
```

* **Output:** `report.json`, `per_step.png`, per-seed figures, and `tifxyz/` (`seedNN_gt` and `seedNN_<model>`) for VC3D.
* **Time:** about 20 to 40 minutes for 4 to 6 models at 30 and 60 steps.
* **Comparing:** compare only models benchmarked on the same machine. The GB10 and the H100 differ slightly in bf16,
  which flips a few threshold decisions over a trace, so local and H100 numbers must not be mixed. Each machine is
  deterministic run to run.
* **Re-scoring:** `trace_benchmark.py --rescore DIR` re-scores existing folders' switch counts from their tifxyz.

**Long traces for inspection:** `trace_spread.py --checkpoint ... --steps 120 --tag NAME --path-map ...` traces
seeds spread over z and writes them to `output/spread/NAME/`.

**Mine failure spots:**
`hard_spots.py --checkpoint ... --seeds 1500 --out FILE.json --path-map ...` takes about 20 minutes. It writes
progress every 100 traces, so it can be stopped early.

## 4. Training

```bash
python -m vesuvius.neural_tracing.sheet_tracing.train --config CONFIG.json --name RUN [--init CKPT] [--set section.field=value ...]
python -m vesuvius.neural_tracing.sheet_tracing.train --resume output/RUN/last.pt
```

* **`--init`:** starts from a checkpoint's weights (its EMA weights if present) with a fresh optimizer and schedule.
  Modules the checkpoint lacks start at zero, so the model begins unchanged.
* **`--resume`:** continues a run, including the optimizer and the elapsed time.
* **Configs:** in `configs/`. The H100 ones are `configs/remote_*.json`; the current recipes are
  `remote_m7_fix.json` and `remote_m7_hard.json` (300 minutes, batch 64).
* **Option reference:** README.md.

The patch index is cached in `output/cache/`. It is rebuilt automatically when the patch directories or
`data/patches.py`'s `INDEX_VERSION` change; it measures each patch's real cell size where `meta.json` is wrong.

## 5. The H100 machine

* **Login:** `ssh ubuntu@147.224.219.64`. It's a shared machine, so other people's jobs run on it.
* **GPU: use GPU 7 only.** GPU 6 is in use by others. A training run (batch 64) takes about 44 GB of GPU 7's 80 GB
  and about 1 to 1.2 s per iteration. A run can share the GPU with a 20 to 25 GB job, but shared runs go slower, and
  runs are capped by time, so they finish with fewer iterations.
* **Everything stays under `/ephemeral/sean/sheet_tracing`**, including temporary files and caches. The root disk
  is shared and nearly full: do not write anywhere else (plain logs and text are tolerated, nothing else). Always
  `source /ephemeral/sean/sheet_tracing/env.sh` first. It points every temp and cache directory (tmp, uv, pip,
  matplotlib, torch, triton, inductor, numba, CUDA) into `cache/`, sets `PYTHONPATH` to the repo copy and `$PY` to
  the venv, and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* **Other people's resources:** keep DataLoader workers at about 8 to 10 per run, and use batch 64 at level 1 (lower
  at level 0). Large batches with many workers have used about 190 GB of shared memory, which can break other
  people's runs.

**Layout** of `/ephemeral/sean/sheet_tracing`:

| path | what |
| --- | --- |
| `villa/vesuvius/src/vesuvius/neural_tracing/sheet_tracing/` | the code (a copy of this package); `output/` holds the runs, `configs/` the configs |
| `data/` | `s1_ds2.zarr` (levels 0 and 1 decompressed in place), `m7_ds2.zarr`, `fiber_presence.ome.zarr` (array `3` decompressed), both patch sets, `hard_windows_m7_len2.json` |
| `logs/` | `RUN.log` (training output, tqdm with `\r`) and `RUN.pid` |
| `env.sh`, `launch.sh` | environment, and the detached launcher |
| `cache/` | all temp and cache files |
| `schedule_round*.sh` | old job chains (finished; examples of waiting on a run with `kill -0 $(cat logs/RUN.pid)`) |

**Sync the code** after changing it here (configs are created on the H100, so they are excluded):

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
$B/launch.sh 7 remote_m7_fix.json my_run --init $O/ctx2_m7_len2/last.pt   # GPU, config, run name, train args
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
  `last.pt` is saved every 1,000 iterations.
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

Then benchmark it locally with `scripts/bench_local.sh` against the models in section 1, and record the result in
`output/remote/RESULTS.md`.
