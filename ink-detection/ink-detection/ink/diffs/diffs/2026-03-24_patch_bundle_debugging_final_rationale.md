Patch-Bundle Debugging Final Rationale
Date: 2026-03-24

Files:
- `diffs/2026-03-24_patch_bundle_debugging_final.diff`
- `diffs/2026-03-24_patch_bundle_debugging_rationale.md`

This file is the final summary for the patch-bundle debugging round. It supersedes the earlier raw diff snapshot when you want the exact code that matched the later passing Slurm test run and the later successful patch-bundle runs.

What was wrong:

1. `erm_patch_bundle` did not auto-generate bundles.
- The first direct run failed immediately because the manifest files under `.tmp/patch_bundles/...` did not exist yet.
- Root cause: the experiment used a plain `PatchBundleDataRecipe`, which expects prebuilt bundle manifests on disk.

Fix:
- `ink/experiments/erm_patch_bundle.py`
- `ink/recipes/data/patch_bundle/recipe.py`
- `ink/recipes/data/patch_bundle/__init__.py`
- Switched the experiment to `GeneratedPatchBundleDataRecipe`.
- Added build-time bundle generation through `PatchBundleWriter.ensure(...)`.

Why:
- This makes `python -m ink.experiments.run erm_patch_bundle` usable without a separate manual preprocessing step.

2. Patch-bundle datasets could train, but stitch planning could not treat them like the source Zarr datasets.
- Patch-bundle runs either failed or fell back to incomplete stitch configuration because stitch planning assumed a `ZarrPatchDataset`.
- Root cause: the bundle dataset did not expose enough source-layout context to derive stitch segment specs and per-segment loaders.

Fix:
- `ink/recipes/data/patch_bundle/recipe.py`
- `ink/recipes/stitch/plan_from_zarr.py`
- `ink/recipes/stitch/loaders.py`
- `ink/recipes/stitch/zarr_prep.py`
- Added source-layout metadata to `PatchBundleDataset`: dataset root, layout, label/mask suffixes, caches, `data_context()`, segment ids, and per-segment sample indexing.
- Added `PatchBundleViewDataset` so train-viz and eval-style loaders can be built from bundle-backed samples.
- Extended stitch runtime prep and stitch loader construction to accept `PatchBundleDataset` as well as `ZarrPatchDataset`.

Why:
- Without this, stitch evaluation could not derive segment specs from patch bundles, and train-viz stitch loaders ended up missing or mismatched.

Concrete failures fixed:
- `stitched evaluation requires stitch.eval.segments`
- `train stitch loaders/segment_ids length mismatch (0 vs 2)`

3. Warm patch-bundle reuse was still extremely slow.
- Even after a bundle already existed, warm runs still spent a long time before entering training or finishing stitched eval.
- Root cause: the code still needed expensive source discovery work on the source Zarrs during bundle reuse and stitch derivation.

Fix:
- `ink/recipes/data/layout.py`
- `ink/recipes/data/patch_bundle/writer.py`
- `ink/recipes/stitch/plan_from_zarr.py`
- Added `segment_source_metadata_fingerprint(...)` to fingerprint Zarr metadata instead of hashing full source contents.
- Bumped bundle schema to `3`.
- Persisted `segment_specs` directly into the bundle manifest.
- Added a manifest-backed fast path so stitch derivation can reuse those persisted segment specs instead of rereading the original segment masks.

Why:
- Full content hashing and mask-based rediscovery were the main reason warm reuse still looked “hung”.
- Persisting the already-derived segment specs turns warm stitch planning into metadata lookup instead of source re-analysis.

Observed effect:
- Earlier warm patch-bundle reuse still spent roughly `100s+` in stitch derivation on the auto-grown validation segment.
- After schema-3 manifests with stored `segment_specs`, warm derive time dropped to effectively `0.000s` in the debug runs.

4. Cold bundle builds were hard to reason about.
- During cold creation there were long pauses with little signal about where time was going.

Fix:
- `ink/recipes/data/patch_bundle/writer.py`
- Added progressive bundle write logging and timing around fingerprint/build stages.

Why:
- This made it possible to separate “writing bundle data” from “fingerprinting source state” from “stitch derivation”.

5. There was no small patch-bundle experiment to debug quickly.
- Debugging only on the full experiment made turnaround slow and expensive.

Fix:
- `ink/experiments/erm_patch_bundle_debug_autogrown_w00.py`
- Added a small patch-bundle debug experiment mirroring the existing two-train / two-val Zarr debug split.

Why:
- This made it possible to reproduce failures faster, validate warm reuse behavior, and confirm that run directories/checkpoints were being written correctly.

6. The new behavior needed regression coverage.

Fix:
- `tests/test_data_recipe_behavior.py`
- `tests/test_stitch_recipe_behavior.py`
- `tests/test_erm_experiment_behavior.py`
- Added or updated tests for:
  - generated bundle creation
  - metadata-fingerprint invalidation behavior
  - deriving stitch runtime from patch bundles
  - patch-bundle train-viz loader construction
  - the debug patch-bundle experiment shape

Why:
- The failures were integration failures across data recipe, stitch planning, and experiment config. Unit coverage alone was not enough.

7. Slurm wrappers were added for reproducible validation.

Files:
- `run-ink-tests.sh`
- `run-ink-wip-erm-patch-bundle.sh`
- `run-ink-wip-erm-patch-bundle-debug.sh`

Why:
- The environment is Slurm-first, and using the same runtime path for tests and experiment launches removed ambiguity while validating the fixes.

What these changes achieved:
- `erm_patch_bundle` can now generate its own bundle when missing.
- The patch-bundle debug experiment trains cleanly.
- Warm patch-bundle reuse no longer re-derives stitch geometry from source masks.
- Run directories are populated with `history.jsonl`, `summary.yaml`, `eval/latest.yaml`, and checkpoint files after epoch completion.
- The final code state passed the Slurm test job that ran `229` tests with `5` skips.

Useful validation points from the successful runs:
- Cold schema-3 debug run rebuilt the bundle and then derived train/eval stitch segments in `0.000s`.
- Warm reuse debug run logged bundle reuse and again derived train/eval stitch segments in `0.000s`.
- Run directories for those debug bundle runs contain epoch outputs, eval summaries, and checkpoint files.
