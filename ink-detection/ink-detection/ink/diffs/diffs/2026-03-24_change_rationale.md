# 2026-03-24 Change Rationale

This file explains why each change bundle was made, what evidence led to it, and what it improved.

## Outcome

The stitched-eval cache-hit path is much better now.

- Old cache-hit debug run `3937`: `item_discovery_s=189.394`
- New metadata-fingerprint path after warm cache in `3940`: `item_discovery_s=0.001`
- Full `erm` run `3941` after warm cache: `item_discovery_s=0.001`

The main stitched-eval bottleneck moved from "minutes of cache-key discovery" to a few seconds of normal finalize/output work.

## Diffs And Why

### `2026-03-24_batch24_log1_runner_profiling.diff`

What changed:

- Restored CUDA/device selection in the `ink` runner.
- Restored one-off env overrides for batch size and `log_every_n_steps`.
- Added per-step timing logs so the trainer shows `data_s` and `step_s`.

Why:

- A copied `ink/experiments/run.py` had dropped the device argument, which caused Slurm runs to execute on CPU by accident.
- We needed to force `batch_size=24` and `log_every_n_steps=1` without permanently changing experiment defaults.
- We needed enough terminal detail to separate "dataloader wait" from "compute step" time.

Evidence:

- Before the runner fix, the jobs were spending long periods on host-side work and never showed the expected GPU behavior.
- After the runner fix, debug runs reached steady-state `step_s` around `0.18-0.22s` after first-batch warmup on GPU.

### `2026-03-24_runfs_save_logging.diff`

What changed:

- Added explicit logs when train history, eval summary, latest eval, and checkpoints are written.

Why:

- We needed to verify that run directories were being populated after each epoch, not only at the end.
- Earlier ambiguity came from runs started before `RunFS` was attached by the CLI entrypoint.

Evidence:

- Logs now show lines like:
  - `[runfs] epoch=1/30 split=train ... history_path=...`
  - `[runfs] epoch=1/30 split=val ... summary_path=... latest_eval_path=...`
  - `[runfs] epoch=1/30 checkpoint=last path=...`
- Run directories now clearly contain:
  - `history.jsonl`
  - `summary.yaml`
  - `eval/latest.yaml`
  - `checkpoints/last.pt`

### `2026-03-24_eval_finalize_profiling_and_cache.diff`

What changed:

- Added timing around stitched-eval finalize and related stages.
- Added more visibility into cache use during eval.

Why:

- The visible "hang" after `[train] start` and during eval needed to be decomposed into dataset build, item discovery, stitching, and output write phases.
- Without subphase timings, it was unclear whether the stall was in component discovery, metric computation, or artifact writing.

Evidence:

- Logs started showing specific finalize timings such as:
  - old path: `finalize stitch done dt_s=221.163`
  - optimized debug path: `finalize stitch done dt_s=4-7s`
  - optimized full path: `finalize stitch done dt_s=2.509`

### `2026-03-24_stitch_component_discovery_profile.diff`

What changed:

- Added detailed profiling around stitched component discovery.

Why:

- Disk cache hits were still slow, which implied the cost was not loading the cached `.npz` itself.
- We needed to isolate whether the delay came before cache lookup, during lookup, or after loading.

Evidence:

- Old cache-hit run `3937` showed:
  - component cache load around `0.08s`
  - but `item_discovery_s=189.394`
- That proved the cache files themselves were not the problem.

### `2026-03-24_stitch_component_subphase_logging.diff`

What changed:

- Added subphase logging for:
  - full supervision read
  - ROI detection
  - ROI label/component extraction
  - save time

Why:

- We needed to measure the miss path precisely on the large auto-grown validation segment.

Evidence:

- Diagnostic run `3933` showed:
  - `supervision_only_read_s=80.615`
  - `labels_only_read_s=84.528`
  - `label_and_supervision_read_s=171.931`
  - `roi_detect_s=4.528`
  - `roi_label_read_and_component_detect_s=9.212`
- This proved the miss path was wasting a full label read across the whole segment.

### `2026-03-24_stitch_component_roi_label_optimization.diff`

What changed:

- Split stitched-eval reading into:
  - full supervision read for ROI detection
  - ROI-only label reads for component extraction
- Added separate reader methods for label and supervision access.

Why:

- The old miss path always read both label and supervision for the full segment before knowing where the interesting regions were.
- For large segments, this was unnecessarily expensive because labels were only needed inside the detected ROIs.

Evidence:

- `3933` showed the full label read cost around `84s`.
- In the first post-change miss under the new fingerprint path, `3940` showed:
  - `segment=auto_grown_... cache=disk_miss ... read_s=91.676 roi_s=4.304 component_s=10.318 dt_s=106.577`
- That is still expensive on a cold miss, but no longer includes the extra whole-segment label pass that had pushed the old path much higher.

### `2026-03-24_stitch_component_metadata_fingerprint.diff`

What changed:

- Changed stitched-eval cache fingerprinting to use metadata-based signatures instead of hashing chunk contents on the hot path.
- Kept the more robust full fingerprint code as a fallback.

Why:

- The remaining cache-hit slowdown was happening before cached component files were loaded.
- The hot path was computing a portable source fingerprint by walking and hashing label/supervision chunk contents, which effectively re-read the zarr trees on every eval.

Evidence:

- Old cache-hit run `3937`:
  - `item_discovery_s=189.394`
  - cached component load itself only about `0.08s`
- First run under the new key scheme in `3940` rebuilt the caches once:
  - epoch 1: `item_discovery_s=291.867` because both segments were `cache=disk_miss`
- Once the new-key caches were populated, the same run dropped to:
  - epoch 2 onward: `item_discovery_s=0.001`
- Full `erm` run `3941` also reached:
  - `item_discovery_s=0.001`

### `2026-03-24_stitch_component_cache_and_test_alignment.diff`

What changed:

- Aligned tests and cache behavior with the stitched-eval changes.

Why:

- The eval and fingerprint changes affected cache expectations and test assumptions.
- We needed the test suite to remain the source of truth while iterating on performance fixes.

Evidence:

- Slurm test jobs completed successfully after these changes.

## Run Directory Verification

These runs prove the save path is working:

- Debug run:
  - `runs/2026-03-24_erm_debug_autogrown_w00_c5f300`
  - contains `history.jsonl`, `summary.yaml`, `eval/latest.yaml`, `checkpoints/last.pt`
  - contains stitched eval artifacts for every epoch and stitched train artifacts on the configured cadence
- Full run:
  - `runs/2026-03-24_erm_a6dd05`
  - contains `history.jsonl`, `summary.yaml`, `eval/latest.yaml`, `checkpoints/last.pt`

## Current Status

What is better now:

- CUDA execution was restored.
- Per-step logging is available.
- Run directories are populated each epoch.
- Stitched-eval cache-hit discovery time is effectively eliminated after warmup.

What is still not fixed:

- W&B step-order warnings remain in the latest completed runs:
  - `Tried to log to step X that is less than the current step Y`
- That is now the main correctness/telemetry issue still visible in the logs.
