# Plan

Inspect every JSON with the existing fiber parser and establish volume identity
from embedded provenance, not directory spelling. Preserve both Paris4 collections
and the supplied train/test membership. Derive a standalone config from
train_s1a_nml_8gpu_fast.json: retain SD1, BF16, GroupNorm, output stride 4,
128-cubed patches, physical target batch 16 and effective batch 128, augmentations,
optimizer/test settings. Replace only datasets/run identity and explicitly disable
embeddings. No NML transforms or normals needed for regular dense testing.

## Spec update
No runtime changes. Preserve managed launch, measured calibration, deterministic
split, and existing sampling/loss semantics. Five scrolls/six source directories;
PHerc1229 is a mislabeled directory for PHerc1299 per every embedded source URL.

## Docs updates
Add dataset mapping/counts, split limitations, launch and calibration prerequisites
to a dedicated training document. Keep earlier uncommitted manager work untouched.

## Tests
CPU only: config parser, all 996 fiber files, matching declared source identities,
no shared resolved files between train/test, no NML/transforms, settings comparison
against baseline. Verify available base metadata/control point bounds if possible.

## Changelog
Record this config in the dedicated task log/document. Dedicated planning folder
avoids overwriting still-uncommitted manager task records (workflow deviation).
