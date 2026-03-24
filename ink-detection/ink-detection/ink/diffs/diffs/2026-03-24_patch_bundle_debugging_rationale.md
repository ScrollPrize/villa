Patch-Bundle Debugging Rationale
Date: 2026-03-24

Files covered by the raw diff:
- `diffs/2026-03-24_patch_bundle_debugging.diff`

Why each change was made:

1. `ink/recipes/data/patch_bundle/recipe.py`
- Added `GeneratedPatchBundleDataRecipe` usage support so `erm_patch_bundle` and the new debug bundle experiment can generate missing bundles automatically before loading them.
- Extended `PatchBundleDataset` with the source-layout metadata needed by stitch planning: `dataset_root`, `layout`, label/mask suffixes, caches, and a `data_context()` bridge.
- Added `PatchBundleViewDataset` and segment-index helpers so patch-bundle train-viz loaders can be built per segment with eval transforms.
- This fixes two concrete failures:
  - initial `FileNotFoundError` because no manifest existed yet
  - epoch-1 crash from `train stitch loaders/segment_ids length mismatch (0 vs 2)`

2. `ink/recipes/data/patch_bundle/writer.py`
- Added progressive bundle write logging so long cold builds are observable.
- Switched bundle source fingerprinting from full content hashing to metadata-based hashing and logged `fingerprint_s`.
- Added `segment_specs` to bundle manifests and bumped the bundle schema to `3`.
- Evidence:
  - old warm patch-bundle runs spent about `177s` then about `97s` in stitch derivation after reuse
  - after manifest-backed `segment_specs`, stitch derivation on the rebuilt schema-3 bundle dropped to `0.000s`

3. `ink/recipes/data/layout.py`
- Added `segment_source_metadata_fingerprint()` so patch-bundle fingerprints can avoid rehashing full source trees.
- This removes the silent post-write pause after `[bundle] progress ... samples=.../...`.

4. `ink/recipes/stitch/plan_from_zarr.py`
- Taught stitch derivation to work with `PatchBundleDataset`, not just `ZarrPatchDataset`.
- Added timing logs for overall stitch derivation and per-segment timings during investigation.
- Added manifest-backed `segment_specs` fast-path.
- Evidence:
  - without this, patch-bundle runs failed with `stitched evaluation requires stitch.eval.segments`
  - before the manifest-backed fast-path, warm reuse still spent about `6s` on train derivation and about `99s` on eval derivation, dominated by `auto_grown_20250919055754487_inp_hr_2um`

5. `ink/recipes/stitch/loaders.py`
- Added patch-bundle support for train-viz stitch loaders.
- This fixes the run-time crash after epoch 1 where patch-bundle training had stitch segment ids but zero loaders.

6. `ink/experiments/erm_patch_bundle.py`
- Switched the experiment to the generated-bundle recipe so `python -m ink.experiments.run erm_patch_bundle` works without manual pre-generation.

7. `ink/experiments/erm_patch_bundle_debug_autogrown_w00.py`
- Added a small patch-bundle debug experiment mirroring the existing 2-train / 2-val debug Zarr experiment.
- This was necessary to iterate quickly and isolate bundle and stitch costs before using the full experiment.

8. Tests
- `tests/test_data_recipe_behavior.py`
  - added coverage for generated bundle creation
  - aligned stale-bundle testing with metadata fingerprinting by mutating Zarr metadata instead of raw chunk contents
- `tests/test_stitch_recipe_behavior.py`
  - added coverage for deriving stitch runtime from a patch bundle
  - added coverage for patch-bundle train-viz stitch loaders
- `tests/test_erm_experiment_behavior.py`
  - added coverage for the debug patch-bundle experiment shape

Observed outcomes after the fixes:
- Cold debug bundle build:
  - train split wrote successfully
  - valid split wrote successfully
  - schema `3` manifests now contain `segment_specs`
- Warm debug bundle reuse:
  - logs `bundle reuse`
  - stitch derive train `dt_s=0.000`
  - stitch derive eval `dt_s=0.000`
  - train-viz loaders count matches segment count
  - training enters `[train] start ...` cleanly

Remaining live observation:
- `history.jsonl` appears immediately after epoch-1 training.
- `checkpoints/last.pt` and `eval/latest.yaml` appear only after the epoch-1 eval/finalize/save block completes, so they can lag `history.jsonl` while stitched eval is still running.
