# S1A NML All-Data Training Config Plan

## Goal

Make the S1A NML config clearly represent an all-S1A-NML training-data config.

## Implementation Plan

1. Rename `configs/loader_example_s1a_nml.json` to
   `configs/train_s1a_nml_all.json`.
2. Set `training.run_name` to `s1a_nml_all`.
3. Keep the existing full S1A glob
   `/home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fiber_vols/fibers_s1a_*.nml`.
4. Remove the stale JSON `test_datasets` block so this config does not mix
   old JSON held-out fibers with the S1A NML training source.
5. Update docs/changelog references to the renamed config.

## Spec Update

No semantic loader spec change is required. The existing NML, affine transform,
and deterministic all-data `max_sample_index: 0` behavior already cover this.

## Docs Updates

Update `docs/code_structure.md` to reference `train_s1a_nml_all.json` and state
that it intentionally omits `test_datasets`.

## Changelog

Update the existing S1A NML config entry to use the new config name and all-data
wording.

## Validation

- Parse the JSON config.
- Resolve the dataset glob and verify it matches all 8 local S1A NML files.
- Load the config with `load_config` to verify schema compatibility.
