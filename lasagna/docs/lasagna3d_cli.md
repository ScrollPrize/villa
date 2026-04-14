# `lasagna3d` CLI

`lasagna3d` is the analysis/tooling CLI for lasagna 3D training. It lives as
a small Python package at `lasagna/lasagna3d/` and is invoked via
`python -m lasagna3d`. New analysis subcommands should be added here rather
than as loose scripts under `lasagna/`.

## Install / run

```bash
PYTHONPATH=vesuvius/src/:lasagna/ python -m lasagna3d <group> <command> [args]
```

The `PYTHONPATH` prefix makes the vendored `vesuvius` library importable and
puts `lasagna/` on the path so `lasagna3d` can import sibling modules such as
`tifxyz_lasagna_dataset`.

## Subcommands

### `dataset vis` — inspect training samples

Renders samples from `TifxyzLasagnaDataset` as JPEGs showing the three
mid-planes of the 3D patch with per-surface contour overlays and chain labels.

```bash
PYTHONPATH=vesuvius/src/:lasagna/ python -m lasagna3d dataset vis \
    --train-config lasagna/configs/tifxyz_train_s3.json \
    --vis-dir ./tmp/dataset_vis \
    --num-samples 10
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--train-config` | *required* | Path to a lasagna training JSON config (same schema as `train_tifxyz.py`). |
| `--vis-dir` | *required* | Output directory; created if missing. |
| `--num-samples` | `10` | Samples to render **per dataset** in the config. |
| `--seed` | `0` | Seed for the per-dataset deterministic shuffle. |
| `--patch-size` | config value | Override `patch_size` from the config. |
| `--num-workers` | `os.cpu_count()` | DataLoader workers for parallel per-sample extraction *and* thread-pool size for parallel render/save. `0` runs everything on the main thread. |
| `--inference-tile-size` | `None` | Cubic patch size of the CT crop the model is run on (single forward). The **dataset is unaffected** — it always emits the training config patch size, with the same patches, GT, and surface masks training would see. Only the CT input fed to the model is read at this size, centered on the same patch's world bbox center (zero-padded outside the volume). Loss and residuals are computed at `min(config_patch_size, inference_tile_size)` (center-cropped from both pred and target). Vis is rendered at the config patch size; `pred` is center-cropped or zero-padded to that size for display. Defaults to the config patch size. |
| `--model` | `None` | Optional checkpoint path. When set, runs inference on the same image the dataset hands to training, computes per-channel losses (same `ScaleSpaceLoss3D` instances training uses), and adds rows for prediction + residuals. Loss values are printed in the figure title. **The checkpoint's `patch_size` is read first and overrides both `--patch-size` and the dataset config's `patch_size`** — `NetworkFromConfig.autoconfigure` derives the encoder stage count from patch size, so any mismatch would silently load a wrong architecture. The checkpoint is loaded **strictly**: any missing or unexpected state-dict key raises `RuntimeError`. For old checkpoints that don't embed `patch_size`, the loader falls back to a sibling `config.json` (the file `train_tifxyz.train` writes alongside checkpoints). |
| `--same-surface-threshold` | config value | Voxel-p25 distance threshold for the same-surface merge inside `compute_patch_labels`. Duplicate wraps (consecutive in chain, different source segments, unsigned 25th-percentile distance ≤ threshold) are collapsed into one surface for EDT, validity bracketing, and all derived losses — the vis still draws both contours but they share a label and color. When unset, falls back to the training config's `same_surface_threshold` field (default `None` = off; recommended `2.0`). See `lasagna/tifxyz_labels.py:detect_same_surface_groups`. |

The command iterates each entry in `config.datasets`, builds a
single-dataset `TifxyzLasagnaDataset`, deterministically shuffles its
patches (`random.Random(seed + dataset_idx)`), and renders the first
`--num-samples` as JPEGs. Entries whose `volume_path` is `null` are skipped.

**No duplicated pipeline code.** The vis drives the exact same three
entry points the training loop uses:

1. A `torch.utils.data.DataLoader` with `num_workers=--num-workers`
   walks the chosen indices through `TifxyzLasagnaDataset.__getitem__`,
   so per-sample extraction (zarr reads, voxelization, chain building)
   runs in worker subprocesses in parallel.
2. `collate_variable_surfaces` collates each sample into a batch-of-one.
3. `train_tifxyz.compute_batch_targets` runs on the main thread against
   CUDA to produce `cos`, `grad_mag`, and `validity` — naturally
   serialized on one GPU.

Render + JPEG save is submitted to a `concurrent.futures.ThreadPoolExecutor`
of the same size, so matplotlib's Agg backend runs concurrently with
the next sample's GPU compute and the next-next sample's worker
extraction. Each figure is local to its render task so Agg stays
thread-safe.

The vis never re-implements voxelization, normals, chain building,
label derivation, or scale-space pooling. The dataset is opened with `include_geometry=True`,
which asks `TifxyzLasagnaDataset` to emit a `surface_geometry` field
alongside the training tensors; `surface_geometry[i]` is the raw
`{points_local, normals_zyx}` that was already computed to produce
`direction_channels[i]`. If training changes, the vis follows
automatically.

Each JPEG is a 3-column grid (axial / coronal / sagittal mid-planes) with
multiple rows stacking training inputs, labels, and loss masks so the whole
supervision signal can be eyeballed in one image. Planes are cut at
`z = Z // 2`, `y = Y // 2`, `x = X // 2`.

**Row layout** (rows 3–7 require CUDA — `compute_patch_labels` uses CuPy
EDT; without CUDA only rows 1–2 are drawn and a "no CUDA" placeholder is
rendered for the rest):

| Row | Content |
|---|---|
| 1 | CT slice + per-chain contour overlay + chain labels |
| 2 | CT slice + projected surface-normal arrows (quivers) |
| 3 | `cos` supervision signal, masked by validity, `twilight` colormap, `[0, 1]` |
| 4 | `grad_mag` supervision signal, masked, `viridis`, auto-ranged (vmax = 99th percentile over valid voxels) |
| 5 | Validity mask at full scale |
| 6 | Validity mask at scale 1 (pooled by 2, erosion of `1 − mask`) |
| 7 | Validity mask at scale 2 (pooled by 2 again) |

When `--model` is passed, four additional rows are appended:

| Row | Content |
|---|---|
| 8 | `pred cos`, `gray`, `[0, 1]` |
| 9 | `pred grad_mag`, `viridis`, auto-ranged |
| 10 | Per-voxel `cos` residual `(pred − target)² · mask` at full res, `hot`, auto-ranged |
| 11 | Per-voxel `grad_mag` residual `smooth_l1(pred, target) · mask` at full res, `hot`, auto-ranged |
| 12 | `cos` scale-space residual sum: residuals at every level of the same masked-average pyramid `ScaleSpaceLoss3D` uses, upsampled nearest to full res and summed across scales. Shows where the multi-scale loss is being charged. |
| 13 | `grad_mag` scale-space residual sum (same construction with `smooth_l1`) |

Per-channel loss values (`loss_cos`, `loss_mag`, `loss_dir`, `total`)
are computed with the exact same `ScaleSpaceLoss3D` instances the
training loop uses and added to the figure title.

The scale-space pyramid is built by calling
`tifxyz_labels.scale_space_validity_pyramid`, which is the same helper
`ScaleSpaceLoss3D` consumes per step during training
(`scale_space_pool_validity`). The pooling rule is **any-valid**: a
coarse voxel is valid iff *at least one* of its eight fine children was
valid (i.e. plain `max_pool3d` on the validity mask). This pairs with
the loss's masked-average pooling of the prediction and target — at
each coarser scale, signal is averaged only over the valid voxels of
each 2×2×2 block, so a single valid fine voxel still produces a
meaningful coarse target. Coarser levels are upsampled nearest-neighbor
back to full res so panels line up visually; the coarse native shape is
shown in each subtitle.

Contours are drawn with `skimage.measure.find_contours` on the voxelized
surface mask slice. Contour colour and normal-arrow colour are **per chain**
(not per surface), so wraps belonging to the same chain share a colour
across rows and panels.

Normal arrows are projected onto each plane: surface points whose
out-of-plane coordinate is within ±0.6 voxels of the slice are picked,
subsampled to at most 80 per panel, and drawn as unit-length quivers
(~6 px). The in-plane components of the ZYX normals are used directly
(axial: `nx`, `ny`; coronal: `nx`, `nz`; sagittal: `ny`, `nz`).

### Chain labeling

Chains are built via `tifxyz_lasagna_dataset.build_patch_chains()`, which
ports the triplet neighbor logic from
`vesuvius/neural_tracing/datasets/dataset_rowcol_cond._build_triplet_neighbor_lookup`.
Per-patch:

1. For each wrap, compute a 2D median `(x, y)` of its stored-resolution
   coordinates.
2. Pick the ordering axis with the larger spread as the local
   through-the-scroll direction.
3. Sort wraps along that axis and link each wrap to its nearest compatible
   neighbor on each side. "Compatible" (`_triplet_wraps_compatible`) means:
   same segment **or** filename-derived wrap ids differ by exactly 1
   (consecutive `w<N>` winding tokens in the segment path or uuid).
4. Walk reciprocal next-links from chain heads (wraps whose prev does not
   reciprocate) to form chains. Asymmetric leftovers are collected as
   singleton chains.

Labels are `{chain_letter}{pos}` — `a0`, `a1`, `a2`, `b0`, `b1`, ... Each
label is drawn at the centroid of that surface's visible pixels in the plane.
A wrap is **complete** when it has both a prev and next neighbor in its chain,
rendered as a **fat** label (fontsize 11, bold). Incomplete wraps (chain
endpoints or singletons) get a **thin** label (fontsize 7, regular).

The title above each JPEG reports the dataset name, patch index, total wrap
count, chain count, and the world bbox.

### `dataset overlap` — per-patch surface-overlap diagnostics

Scans training patches and, for each patch, measures how close
candidate surface pairs get to intersecting. Writes one JSON line per
patch with the worst pair's signed-distance statistics so overlapping
segments in existing datasets can be flagged quantitatively.

```bash
PYTHONPATH=vesuvius/src/:lasagna/ python -m lasagna3d dataset overlap \
    --train-config lasagna/configs/tifxyz_train_s3.json \
    --out ./tmp/overlap.jsonl \
    --num-samples 200
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--train-config` | *required* | Path to a lasagna training JSON config. |
| `--out` | *required* | Output JSONL path; parent directory is created. |
| `--num-samples` | all | Patches to scan per dataset (deterministic shuffle). |
| `--seed` | `0` | Per-dataset shuffle seed. |
| `--num-workers` | `os.cpu_count()` | DataLoader workers. `0` runs everything on the main thread. |
| `--vis-dir` | `None` | If set, render a `dataset vis`-style JPEG per emitted patch into this directory (reuses `dataset_vis._render_sample_figure`, so rows, contours and normal arrows are identical to what `dataset vis` would produce for that patch). Filenames are sortable by worst-pair `p1` so `ls` puts the worst patches first. The title embeds the worst pair's labels and its post-flip `p1` / `p50` / `min` / `n_samples`. Requires CUDA for the label rows. |
| `--vis-top-k` | `None` | With `--vis-dir`, render only the K worst patches after scanning completes (ranked by post-flip `p1`, tie-breaks by `0.1`, then `min`). Without it, every emitted patch is rendered inline during the scan. |

The overlap counter column and the vis merge both use a fixed
rule: a pair is considered "same surface" when its post-flip
signed `p25` is `≤ 2.0`. Pairs flagged this way are unioned into
groups via union-find and passed as explicit `same_surface_groups`
into `compute_patch_labels`, so the vis shows exactly what a
training run with `same_surface_threshold: 2.0` would merge. The
JSONL output is still the raw unmerged stats. There is no
`--same-surface-threshold` flag on `dataset overlap`.

#### Candidate pair rule

A pair `(i, j)` of surfaces in a patch is considered iff:

1. They share the same `chain` in `surface_chain_info`,
2. `|pos_i - pos_j| == 1` (consecutive in that chain), and
3. `segment_idx_i != segment_idx_j` (different source segment files).

These are the only pairs that can reasonably duplicate each other
given how `build_patch_chains` links wraps.

#### Signed distance sampling

For each candidate pair `(A, B)` the command computes a 3D
Euclidean distance transform of `~mask_B` on CUDA with
`return_indices=True` (via CuPy). Each raw surface point `P` on `A`
(from `surface_geometry[A].points_local`, the same tensor the dataset
already splats to produce direction channels) is queried against
`dist_B` and the feature-transform indices give the nearest voxel
`P'` on `B`. Sign comes from `A`'s own normal at `P`:

```
d_signed(P) = |P - P'| · sign((P - P') · n_A(P))
```

Intersection appears as negative samples in the lower tail of the
distribution.

#### Stats and sign flip

For the signed samples of each pair we compute `min`, `max`, and
percentiles at `[0.1, 1, 5, 15, 25, 50, 75, 85, 95, 99, 99.9]`.
If the raw median is negative, all samples are negated before stats
are computed so the **reported median is always ≥ 0**. The record
field `sign_flipped` records whether that happened. Ranking between
pairs and across patches still uses the post-flip `p1` — intersection
severity shows up as a negative `p1` (and possibly `0.1`, `min`).

Per patch the command picks the pair with the lowest `p1`
(tie-break: `0.1`, then `min`) and writes a record of the form:

```json
{
  "dataset": "s3",
  "dataset_idx": 0,
  "patch_idx": 1742,
  "world_bbox": [z0, z1, y0, y1, x0, x1],
  "num_surfaces": 6,
  "num_pairs_considered": 3,
  "num_pairs_evaluated": 3,
  "worst_pair": {
    "a": {"surface_slot": 2, "wrap_idx": 2, "label": "a1",
          "segment_idx": 11, "segment_path": "…/w12_seg_a.tifxyz"},
    "b": {"surface_slot": 3, "wrap_idx": 3, "label": "a2",
          "segment_idx": 14, "segment_path": "…/w12_seg_b.tifxyz"},
    "n_samples": 8421,
    "min": -3.2, "max": 17.8,
    "percentiles": {"0.1": -2.9, "1": -1.7, "5": 0.2, "15": 1.1,
                    "25": 2.4, "50": 4.6, "75": 7.2, "85": 8.9,
                    "95": 12.1, "99": 15.4, "99.9": 17.1},
    "sign_flipped": false
  }
}
```

Patches with zero candidate pairs (e.g. a single surface, or all
surfaces from the same source segment) are silently skipped.
After all datasets finish, a top-20 ranking by `p1` is printed to
stdout so worst-offending patches are easy to spot and re-render with
`dataset vis`.

## Adding new subcommands

The package layout is:

```
lasagna/lasagna3d/
├── __init__.py
├── __main__.py       # entrypoint: python -m lasagna3d
├── cli.py            # argparse dispatch
├── dataset_vis.py    # `dataset vis` implementation
└── ...               # future subcommands
```

To add a new subcommand:

1. Add a parser in `cli.py` under the appropriate group (or create a new
   group).
2. Put the implementation in its own module under `lasagna3d/`.
3. Wire it up by routing `args.group` / `args.command` in `main()`.

Long-lived analysis code should land here instead of drifting as standalone
scripts under `lasagna/`.
