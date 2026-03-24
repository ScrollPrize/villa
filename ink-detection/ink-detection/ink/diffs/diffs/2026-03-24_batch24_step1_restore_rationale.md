# 2026-03-24 Batch 24 / Step-1 Restore

This note records the temporary comparison changes added back after the local code reset.

## What changed

- Restored env-driven override support in `ink/experiments/run.py`
  - `INK_BATCH_SIZE`
  - `INK_LOG_EVERY_N_STEPS`
- Restored runner startup logging in `ink/experiments/run.py`
  - experiment name
  - resolved device
  - effective train/valid batch size
  - effective `log_every_n_steps`
  - run directory
- Restored per-window train timing in `ink/recipes/trainers/patch.py`
  - `it_s`
  - `data_s`
  - `step_s`
- Restored stitched-eval timing logs in:
  - `ink/recipes/eval/validation.py`
  - `ink/recipes/eval/stitch_stage.py`
  - `item_discovery_s`
  - finalize stage durations
  - per-item stitched metric timing
- Updated the ResNet3D default-path test to assert the stable contract
  - standalone repo path
  - under `weights/`
  - ends with `r3d50_KM_200ep.pth`
 - Updated the eval-logging cadence test to assert only on progress lines

## Why

- The experiment files should stay as the authored defaults.
- We still need one-off Slurm comparisons with `batch_size=24` and per-step logging without editing the experiment definitions directly.
- The `data_s` / `step_s` split is useful for answering whether a run is bottlenecked on:
  - waiting for the next batch from the loader
  - or doing model/optimizer work after the batch arrives
- The stitched-eval logs are needed to separate:
  - eval batch iteration
  - item discovery
  - stitched finalize
  - stitched probability writing
- The old model-path test was tied to a previous repo-name substring instead of the actual contract the recipe guarantees now.
- The old eval-logging test assumed every eval log line was a progress line; that stopped being true once finalize timing logs were added.

## Intended use

Submit the standard Slurm wrappers with exported env vars, for example:

- `sbatch --export=ALL,INK_BATCH_SIZE=24,INK_LOG_EVERY_N_STEPS=1 run-ink-wip.sh`
- `sbatch --export=ALL,INK_BATCH_SIZE=24,INK_LOG_EVERY_N_STEPS=1 run-ink-wip-erm-debug.sh`

This keeps the wrapper scripts and experiment recipes simple while making the comparison settings explicit in the job submission and in the run logs.
