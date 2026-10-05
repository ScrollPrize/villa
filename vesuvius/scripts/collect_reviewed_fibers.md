# Reviewed annotation training sets

From the villa checkout, run:

```sh
python3 vesuvius/scripts/collect_reviewed_fibers.py
```

Uses Python's standard library. Fetches the current directory listings and every
JSON under https://dl.ash2txt.org/other/annotators/ (no stale download cache).
Defaults use this host's original collection and train/test split under
`/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/`.
Output is a new timestamped `*_reviewed_annotators_*` directory under
`/ephemeral/hendrik/fiber3d_train/`. Override with `--output`, `--originals`,
`--split`, or `--source`; existing outputs are never overwritten.

Only files tagged `reviewed` are copied to `train/PHerc*/`. This includes
`reviewed` files also tagged `rework` or `tricky`, which remain tagged and are
reported. Do not use a recursive glob across the dataset root: `raw_reviewed/`
contains untouched downloaded copies for provenance, not training inputs.

Parent matching removes the crop's numeric prefix and `_runN` suffix, then
resolves within the original scroll folder. PHerc1299 maps to the legacy
PHerc1229 source folder. Exact source coordinate spaces must match. Missing
parents, unknown H/V, ambiguous train/test membership, held-out parents, invalid
JSON, and output-name collisions fail the whole publication without a partial
training dataset. No geometry-based label fallback exists.

The parent's effective `manual_tag` (if set), otherwise `automatic_tag`, is put
into the piece's **manual_tag**. Piece-local automatic scores remain intact.
This preserves the inherited effective H/V through VC3D recalculation; copying
parent geometric statistics into piece-local automatic fields would not.
`manifest.json` records both classifications, source URLs, parent paths and
source/parent/output checksums. No other annotation fields are modified.

Reruns create a fresh complete snapshot, including newly reviewed files and
excluding files whose review tag was removed. Earlier datasets are untouched.
Directory discovery and output ordering are deterministic; the remote is not
transactionally frozen, so recorded hashes identify the exact bytes fetched.

Tests: `python -m pytest -q vesuvius/tests/scripts/test_collect_reviewed_fibers.py`
