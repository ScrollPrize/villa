# Reviewed-reference cache export

Input evaluations: `reviewed_fibers_20260930_v3` (91 reviewed fibers, three models).

Output root:
`/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/cache_sync_v1`

Each of `lowres`, `fullres`, `gaussian` contains
`vc3d_channels/PHerc0139`, `PHerc0332`, `PHerc1299`, `PHerc1451`, `PHercParis4`.
Each scroll has three Zarr channels and portable `inference.lasagna.json` plus
provenance. Lowres is level3; both fullres models are level1. No resampling.

Scope is **all currently published complete chunks in these model caches**,
including reused coverage from earlier evaluations—not only newly inferred
chunks or a minimal tube around the reviewed fibers. Missing chunks remain
missing; no inference is run. A frozen per-scroll numeric-key inventory records
coverage. The three-channel intersection excludes partially published chunks.

Only chunk files are symlinked; metadata is copied. Hidden `.live_tmp` scratch,
accumulators, checkpoints, traces and writer lock files are excluded. Source
caches are unchanged. Do not sync the enclosing evaluation directory.

For example, copy only the Gaussian model:

```bash
rsync -aL --info=progress2 nat64:/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/cache_sync_v1/gaussian/vc3d_channels/ ./reviewed_gaussian/
```

Use `lowres` or `fullres` instead for the other models, keeping distinct local
destinations. `-L` dereferences the chunk links. Exact logical transfer sizes
(compressed chunk bytes plus metadata, not filesystem allocated size) are in
`cache_sync_v1/export_report.json`.

Those totals cover `vc3d_channels` only, not the small outer inventory/report
files. The inventory freezes key membership, not chunk contents: links still
depend on the original cache files remaining available and unchanged. This is
a link export, not an independent archival copy; `rsync -L` makes the copy.

## Reproduction and validation

Completed export: 1,964,784 verified chunk links across 15 model/scroll pairs.
No broken links. One incomplete lowres Paris4 key excluded; all other selected
keys complete. Total channel transfer size: 169,370,784,242 bytes (157.739 GiB).

| Scroll | Lowres GiB | Previous fullres GiB | Gaussian GiB |
|---|---:|---:|---:|
| PHerc0139 | 6.846 | 27.118 | 0.424 |
| PHerc0332 | 1.140 | 4.633 | 0.013 |
| PHerc1299 | 2.871 | 7.662 | 0.219 |
| PHerc1451 | 0.468 | 1.295 | 0.014 |
| PHercParis4 | 17.259 | 86.507 | 1.270 |
| Total | 28.584 | 127.215 | 1.940 |

The large historical model totals reflect older cached coverage, not data
generated solely by this evaluation. Gaussian is the new evaluated cache.

Set `scripts/training_failure_mining_control.json` action to
`reviewed_cache_export`, then use the approved no-argument command:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
/home/hendrik/.venv_las/bin/python -m pytest -q scripts/test_reviewed_cache_export.py scripts/test_reviewed_fiber_evaluation.py
```

Nine tests cover level1/3, scratch/incomplete exclusions, portable manifests,
immutable inventory resumes and the reviewed cohort. Actual export validates
all links, channel-file counts, source scales/shapes and metadata level paths.
If cache coverage changes after export, choose a new version rather than
silently changing this inventory. No tracing, GPU scheduling or producer
settings changed.
