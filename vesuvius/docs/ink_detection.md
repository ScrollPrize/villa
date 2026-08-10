# Ink Detection

The `vesuvius.ink_detection` package trains and runs per-pixel ink-probability
models on surface volumes. It supports flat 2.5D workflows, native 3D workflows,
and the aligned 21-slice hybrid 3D-to-2D recipe.

## Environment

The package requires Python 3.14. From `villa/vesuvius`, install the model and
test extras:

```bash
uv sync --extra models --extra tests
```

Commands below use the managed environment:

```bash
uv run --extra models python -m vesuvius.ink_detection.training.train -h
```

The `models` extra includes PyTorch, Accelerate, the model-building stack,
OpenCV, TIFF/Zarr codecs, and the CUDA cuCIM package used by positive native-3D
label dilation. The volume-only installation does not include the training
stack.

## Data layout

The published `ink_9um` label tree is scroll-first. A segment and its labels
are colocated:

```text
/data/ink_9um/
├── labels/
│   ├── 0139/
│   │   ├── public_2p4_level2_zmean4/
│   │   │   └── pherc0139-w016/
│   │   │       ├── pherc0139-w016_inklabels.zarr
│   │   │       ├── pherc0139-w016_supervision_mask.zarr
│   │   │       ├── pherc0139-w016_validation_mask.zarr
│   │   │       └── surface-volume.zarr
│   │   └── native_9p362_level0/
│   │       └── w035/
│   │           ├── w035_inklabels.zarr
│   │           ├── w035_supervision_mask.zarr
│   │           └── x.tif, y.tif, z.tif, volume_source.txt, meta.json
│   └── phercparis4/
│       └── public_2p4_level2_zmean4/
│           └── phercparis4-w00/
└── checkpoints/
```

The code does not interpret `ink_9um`, scroll, or family directory names.
`segments_path` and volume paths come from JSON, and labels are discovered next
to each segment. The `sampling_scroll`, `sampling_physical_segment_keys`, and
`sampling_representation_keys` values are identifiers, not paths; their values
do not change when storage layout changes.

Required label basenames are `<segment>_inklabels.zarr` and
`<segment>_supervision_mask.zarr`. A validation mask is optional. Versioned
assets use `_v<N>` before the extension. Without `label_version`, each label
kind independently selects its greatest numeric version. With an explicit
`label_version`, ink and supervision assets must both exist at that version.

## Volume paths and disk cache

Supply each volume path exactly as it should be opened: a local path, `s3://`
URL, or `https://` URL. Ink detection does not rewrite endpoint spellings.
S3 paths containing `vesuvius-challenge-open-data` use anonymous access;
HTTPS volumes may use the username/password JSON selected by
`volume_auth_json`.

The optional on-disk compressed-chunk cache requires Zarr 3. Set
`volume_cache_dir` for training or flat inference, or `--cache-dir` for native
inference. Uncached volume reads remain supported with Zarr 2.18.7 and Zarr 3;
the package dependency range remains `zarr>=2.18.7,<4`. Each authored volume
path receives its own hashed subdirectory, so
`volume_cache_max_gb` and `--cache-max-gb` are decimal-GB budgets per volume,
not totals across the cache root.

When a volume opens with a budget, the code scans that volume's cache
subdirectory once. An over-budget directory is pruned oldest-mtime-first to at
most 90% of the budget; an at- or under-budget directory is left alone. Zarr's
CacheStore then applies a process-local steady-state LRU. Multiple workers may
share the directory safely because LocalStore installs complete values
atomically, but independent process accounting means the directory can briefly
overshoot its budget before a later open-time sweep. Every process constructs
its own source store after it starts; a remote fsspec store is never inherited
as the active transport across a fork.

## Shipped aligned-corpus configs

Two JSON files ship with the package:

```text
src/vesuvius/ink_detection/configs/aligned21_hybrid_3d2d.json
src/vesuvius/ink_detection/configs/aligned21_fixed_scroll_prior.json
```

`aligned21_hybrid_3d2d.json` is a training recipe with placeholder paths. It
uses a 17-of-21 jittered Z window, `robust_mad` normalization, the
`vesuvius_unet_3d_stem_2d` model, smoothed BCE plus Dice, and a fixed batch
prior. Its example dataset follows the scroll-first path
`ink_9um/labels/0139/public_2p4_level2_zmean4/pherc0139-w016/`.

`aligned21_fixed_scroll_prior.json` is the corpus manifest: 29
representations across four scroll identifiers, with per-batch counts
`0139=29`, `1667=22`, `Paris4=11`, and `0814=2`. These counts sum to the
shipped batch size of 64. Copy the relevant representation rows into dataset
entries rather than treating the manifest itself as a training config.
Its `schema_version`, `description`, `strategy`, `seed`, `batch_size`, and
`target_batch_counts` fields describe the sampling recipe. Each item in
`representations` supplies `source_family`, `segment`, `scroll`,
`physical_segment_key`, and `representation_key`; the last three values feed
the corresponding `sampling_*` training fields without becoming path
components.

## Prepare a 9 µm aligned input

Native approximately 9.362 µm surface volumes can be used directly. For a
2.399 µm OME-Zarr, read XY pyramid level 2, select the centered 84 Z planes,
and mean-pool every four planes to 21 slices:

```bash
uv run --extra models python -m \
  vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input \
  /data/raw/pherc0139-w016.ome.zarr \
  /data/ink_9um/labels/0139/public_2p4_level2_zmean4/pherc0139-w016/surface-volume.zarr \
  --level 2 \
  --workers 8
```

The source must be a 3D uint8 array or an OME-Zarr group containing the selected
level. The output is a Zarr-v2 group with one `(21, Y, X)` uint8 array at `0`,
tagged `level2-zmean4-21slice-v1`. Pooling uses float32 means followed by
NumPy half-to-even rounding. The command writes `<output>.partial` and renames
it only after completion. It refuses to run if either the destination or the
partial path exists.

| Argument | Meaning |
|---|---|
| `input_zarr` | Bare 3D array or surface-volume OME-Zarr path/URL. |
| `output_zarr` | New output path; never overwritten. |
| `--level` | Source array key, default `2`. |
| `--workers` | Tile threads, default `8`. |

## Convert label images

After editing label TIFFs or PNGs, convert them beside their segment:

```bash
uv run --extra models python -m \
  vesuvius.ink_detection.preprocessing.create_label_zarrs \
  /data/ink_9um/labels/0139/public_2p4_level2_zmean4 \
  --workers 4
```

The command recursively finds `_inklabels`, `_supervision_mask`, and
`_validation_mask` images with `.tif`, `.tiff`, or `.png` extensions. It writes
a same-stem `.zarr` group with six levels by default. Labels occupy Z slice 32
of a 65-plane volume; XY levels use nearest-neighbor downsampling. A matching
folder max/composite TIFF is also converted, using rounded mean downsampling.
Existing outputs are skipped unless `--overwrite` is present. Per-file errors
are reported and make the command exit nonzero.

| Argument | Meaning |
|---|---|
| `root` | Directory scanned recursively; `.zarr`, `.git`, and `__pycache__` trees are skipped. |
| `--workers` | Parallel conversion workers; default is the smaller of CPU count, file count, and 8. |
| `--levels` | Number of OME-Zarr levels, default `6`. |
| `--overwrite` | Replace existing same-stem `.zarr` outputs. |

## Curation operations

The remaining label and prediction maintenance commands keep the reference
arguments but run below `vesuvius.ink_detection.preprocessing`:

```bash
uv run --extra models python -m vesuvius.ink_detection.preprocessing.validate_segments ROOT
uv run --extra models python -m vesuvius.ink_detection.preprocessing.clean_labels ROOT
uv run --extra models python -m vesuvius.ink_detection.preprocessing.merge_predictions ROOT
uv run --extra models python -m vesuvius.ink_detection.preprocessing.composite_from_zarr --input-root ROOT --method max
uv run --extra models python -m vesuvius.ink_detection.preprocessing.download_required_zarr_chunks --datasets-root DATASETS --volumes-json VOLUMES.json --output-root OUTPUT
```

`validate_segments` checks label binary encodings and TIFF/Zarr spatial shape;
`clean_labels` rewrites matching label images as `{0,255}` TIFFs after optional
component and hole cleanup; `merge_predictions` aggregates matching directional
prediction files; and `composite_from_zarr` writes max or mean TIFF projections.
The downloader uses tifxyz patch geometry to copy only source chunks required
by selected label patches. Use `--dry-run` to inspect its plan before writing.

All long options accept hyphens and underscores interchangeably; help shows the
reference spelling. The commands have these persistent-state guarantees:

- `validate_segments` reports every readable label-content problem even when
  another segment has missing or corrupt metadata. Pass the family directory
  whose immediate children are segment directories as `ROOT`.
- `clean_labels` stages and validates the replacement before moving the source
  into the sibling `label_backup` tree. A failed write leaves the active input
  in place, and a completed rerun is reported as skipped unless `--overwrite`
  is supplied.
- `merge_predictions` retains the reference selection policy: eligible stems
  contain `betti`, `ema`, or `640`, and the greatest parsed checkpoint is used
  for each term and direction. An invocation that writes no merged output fails
  instead of reporting success; decoded floating predictions must be finite.
- `composite_from_zarr` publishes each TIFF only after all projected tiles are
  written. Existing outputs fail unless `--overwrite` is supplied.
- `download_required_zarr_chunks` writes an explicit Zarr-v2 sparse output
  readable by Zarr 2 and 3. Its root attributes record the normalized source,
  every array schema, exact per-scale chunk plan, recompression preset, and
  completed chunk ids. Resume requires all of those plus the actual output
  schema to agree; use `--overwrite` to replace an incompatible store.

The downloader accepts two inert planning flags:
`--stored-grid-pad` and `--patch-finding-workers`. They do not change patch
discovery or the exported chunk plan. Active planning controls are
`--patch-size`, `--overlap-fraction`, `--patch-finding-type`, the subtiling
tile/stride/filter options, `--patch-min-labeled-coverage`, `--patch-filter`,
and `--label-version`. Output controls are `--download-workers`,
`--recompress`, `--dry-run`, and `--overwrite`.

## Train

Copy the shipped hybrid recipe, replace `out_dir` and `datasets`, then run:

```bash
uv run --extra models python -m vesuvius.ink_detection.training.train \
  /path/to/aligned21_hybrid_3d2d.json
```

For two Accelerate workers:

```bash
uv run --extra models accelerate launch --num_processes 2 --module \
  vesuvius.ink_detection.training.train /path/to/aligned21_hybrid_3d2d.json
```

The training CLI has one positional argument, `config_path`. Device placement,
distributed execution, and mixed precision are handled by Accelerate.

### Configuration schema

The JSON object is also stored in every checkpoint. Required values for a new
training run are:

| Key | Contract |
|---|---|
| `description` | Optional human-readable metadata preserved in the checkpoint config. |
| `mode` | `flat`, `full_3d`, or `full_3d_single_wrap`. The last mode adds a surface-mask input channel. |
| `model_type` | `vesuvius_unet`/`unet`, either `_2p5d` form, either `_3d_stem_2d` form, or the `dinov2` compatibility form described below. |
| `model_name`, `autoconfigure` | Optional model-builder name and automatic shape configuration; `model_config.autoconfigure` takes precedence. |
| `model_config` | Model-builder settings. The aligned hybrid uses encoder/decoder blocks, `stem_channels`, and `z_projection_mode`; `input_pad_depth_to` optionally pads model input in Z. |
| `targets` | Must contain only `ink`, with `out_channels: 1` and `activation: "none"`; Z projection is `none`, `max`, `mean`, `logsumexp`, or `learned_mlp`. |
| `in_channels` | Source image channels. Native modes resolve this to 1 or 2 during training. |
| `patch_size` | Positive `[Z, Y, X]` training crop. |
| `patch_overlap` | Training patch-finder stride multiplier, not the inference overlap fraction. |
| `patch_min_labeled_coverage` | Minimum labeled bounding-box area divided by patch area. |
| `datasets` | Non-empty array for labeled training. |
| `num_iterations`, `batch_size`, `seed` | Training length, per-worker batch size, and reproducibility seed. |
| `out_dir` | Checkpoints, metrics, previews, and audit outputs. |

With `model_type: "dinov2"`, set `pretrained_backbone` either at top level or
inside `model_config`; the nested value wins. The config is normalized to the
Vesuvius U-Net builder. `pretrained_decoder_type` is copied into
`model_config` in the same way. A checkpoint path used as a pretrained
backbone is resolved relative to the config-bearing checkpoint during
inference.

Each `datasets` item supports:

| Key | Contract |
|---|---|
| `segments_path` | Required directory containing segment directories. |
| `volume_scale` | Required Zarr level for the dataset. |
| `segments` or `segment_names` | Optional segment allowlist. |
| `surface_volume_paths` | Per-segment flat volume paths; keys may be relative segment paths or names. |
| `surface_volume_path` | One flat volume path, only with exactly one allowlisted segment. |
| `volume_path` | Required native scroll-volume path for `full_3d*` modes. |
| `sampling_scroll` | Scroll identifier required by balanced and fixed-prior sampling. |
| `sampling_physical_segment_keys` | Per-segment physical identity required by fixed-prior sampling. |
| `sampling_representation_keys` | Per-segment representation identity required by fixed-prior sampling. |

Data, patch, and augmentation controls:

| Key | Values and behavior |
|---|---|
| `patch_discovery_mode` | `labeled` (default) or `unlabeled`; the latter reads `unlabeled_datasets`. |
| `patch_finding_type` | `default` or `subtiling`. Subtiling requires `patch_finding_filter_empty_tile: true`. |
| `patch_finding_scale`, `patch_finding_tile_size`, `patch_finding_stride` | Optional subtiling/scan controls. |
| `unlabeled_patch_min_data_coverage` | Rejected: unlabeled discovery uses a fixed `0.25` nonempty-coverage threshold. |
| `label_version` | Optional explicit `v<N>` label version. |
| `patch_cache_filename` | Optional patch-index JSON path; otherwise the cache is under `out_dir`. |
| `volume_auth_json` | Optional HTTPS Basic-Auth JSON with `username` and `password`. Public Vesuvius S3 is opened anonymously. |
| `volume_cache_dir`, `volume_cache_max_gb` | Optional Zarr-3 compressed-chunk cache root and per-volume decimal-GB budget. |
| `augmentation_preset` | `default`, `spatial_only`, `spatial_intensity_no_clip`, or `none`. |
| `augmentation_rotation_axes` | Optional rotation-axis list. |
| `disable_augmentations` | Disable augmentation independently of the preset. |
| `flat_z_window_jitter` | Object with `enabled`, `window_depth`, `max_offset`, `probability`, and `padding: "forbidden"`. |
| `full_3d` | Projection thicknesses plus `label_dilation_distance` and `supervision_dilation_distance`. |
| `normal_pooling.support_grid_max_distance` | Maximum support-grid distance used by native geometry. |

`image_normalization` may be a mode string or an object. Supported canonical
modes are:

| Mode | Additional fields |
|---|---|
| `robust_mad` | `percentile_lower`, `percentile_upper`; defaults 1 and 99. |
| `robust_percentile_span` | `percentile_lower`, `percentile_upper`. |
| `minmax` | None. |
| `percentile_minmax` | `percentile_lower`, `percentile_upper`. |
| `clip_divide` | `clip_min`, `clip_max`, `divisor`; defaults 0, 200, and 255. |
| `clip_zscore` | Required `clip_min`, `clip_max`, `mean`, and positive `std`. |
| `divide` | Positive `divisor`, default 255. |
| `none` | Leaves values as float32. |

Sampling and objective controls:

| Key | Values and behavior |
|---|---|
| `sampling_strategy` | `uniform`, `scroll_segment_balanced`, or `fixed_scroll_prior_stratified`. |
| `fixed_scroll_prior.seed` | Must equal top-level `seed`. |
| `fixed_scroll_prior.target_batch_counts` | Per-scroll integer quotas; their sum must equal `batch_size`. |
| `sampling_audit_every` | Fixed-prior observed-audit cadence; defaults to `save_every`. |
| `loss` | One `LabelSmoothedDCAndBCELoss`, either shorthand fields or a nonempty `terms` array. |
| `loss.bce_label_smoothing`, `loss.dice_label_smoothing` | Label-smoothing factors. |
| `loss.dice_weight`, `loss.ce_weight` | Shorthand Dice and BCE weights. |
| `loss.terms[]` | `name`, optional `metric_name`, `weight`, `weight_dice`, `weight_ce`, smoothing fields, and `bce_kwargs`. |

Optimization, execution, and artifact controls:

| Key | Values and behavior |
|---|---|
| `optimizer`, `learning_rate`, `weight_decay` | Optimizer factory name and base hyperparameters. |
| `optimizer_betas`, `optimizer_momentum`, `optimizer_nesterov` | Optimizer-specific values. |
| `encoder_lr_mult`, `freeze_encoder` | Optional pretrained-encoder LR scaling or freezing. |
| `scheduler.name` | `diffusers_cosine_warmup`, `cosine_annealing`, or `one_cycle`. |
| `warmup_steps` | Diffusers cosine warmup. |
| `scheduler.t_max`, `scheduler.eta_min` | Cosine-annealing fields. |
| `scheduler.max_lr`, `scheduler.total_steps`, `scheduler.pct_start`, `scheduler.final_div_factor` | One-cycle fields. |
| `grad_acc_steps`, `grad_clip`, `max_steps` | Accumulation, clipping, and the optimizer-step horizon used by the scheduler. |
| `mixed_precision` | Accelerate precision setting, such as `fp16`, `bf16`, or `no`. |
| `dataloader_workers`, `pin_memory`, `prefetch_factor` | DataLoader controls. |
| `ddp_find_unused_parameters`, `ddp_broadcast_buffers` | DDP controls. |
| `stitch_factor`, `use_stitched_forward`, `stitched_gradient_checkpointing` | Flat stitched-forward controls; native modes disable stitching. |
| `enable_deep_supervision` | Enable decoder deep supervision. |
| `ema` | `enabled`, `decay`, `start_step`, `update_every_steps`, `validate`, and `save_in_checkpoint`. |
| `val_every`, `val_steps`, `val_preview_batches` | Validation and preview cadence. |
| `save_every`, `log_every` | Checkpoint and log cadence. |
| `best_checkpoint_metric` | `val_loss`, `val_balanced_accuracy`, or null. |
| `verify_finite_gradients_steps`, `max_amp_overflow_events` | Optional gradient/AMP health checks. |
| `checkpoint`, `weights_only` | Resume from a checkpoint path, or load only its model weights. Relative checkpoint paths resolve beside the JSON file. |
| `wandb_project`, `wandb_entity` | Enable W&B. `wandb_entity` is required when `wandb_project` is present. |
| `wandb_resume`, `wandb_run_id` | Resume an existing W&B run; the ID may come from the checkpoint. |
| `benchmark` | Optional `enabled`, `warmup_steps`, and `output_path` object. |

### Training artifacts and calibration

`out_dir` can contain:

- periodic `ckpt_<step>.pth` checkpoints;
- `best_val_loss.pth` or `best_val_balanced_accuracy.pth` and
  `best_checkpoint.json` when best-checkpoint tracking is enabled;
- append-only `validation_metrics.jsonl`;
- `train_previews/` and `val_previews/` LZW TIFF montages;
- `sampling_observed.json` for fixed-prior sampling;
- `gradient_health.json` when finite-gradient checking is enabled; and
- `benchmark_summary.json` when inline benchmarking is enabled.

The online `val_loss` is the configured smoothed training objective. It is not
the calibration-comparable BCE when BCE label smoothing is nonzero. Use
`val_bce_unsmoothed` from `validation_metrics.jsonl` when comparing probability
calibration between runs.

Every checkpoint embeds the canonical config along with model, optimizer,
scheduler, step, and W&B run ID. EMA-enabled checkpoints may also contain
`ema_model` and `ema_optimizer_step`. Inference rebuilds the model and
preprocessing from this embedded config; editing a separate JSON file does not
override a checkpoint.

## Flat inference

Run one surface volume:

```bash
uv run --extra models python -m vesuvius.ink_detection.inference.infer \
  /data/ink_9um/labels/0139/public_2p4_level2_zmean4/pherc0139-w016/surface-volume.zarr \
  /data/ink_9um/checkpoints/hybrid-best.pth \
  /data/predictions/pherc0139-w016.tif \
  --batch-size 4 \
  --gpus 0
```

Run every resolvable segment below a folder:

```bash
uv run --extra models python -m vesuvius.ink_detection.inference.infer \
  --folder /data/ink_9um/labels/0139/public_2p4_level2_zmean4 \
  --checkpoint-path /data/ink_9um/checkpoints/hybrid-best.pth \
  --output-prefix aligned21 \
  --direction both
```

Folder mode writes dated TIFFs below each segment's `preds/` directory and
skips a segment/direction when a matching dated prediction already exists.
Single mode writes the requested path; `--direction both` adds `_reverse` to
the second output.

| Argument | Meaning |
|---|---|
| `input_zarr checkpoint output_tiff` | Single-volume inputs and output. All three are required outside folder mode. |
| `--folder` | One segment directory or a parent of segment directories. |
| `--checkpoint-path` | Explicit checkpoint, useful in folder mode; overrides the positional checkpoint. |
| `--output-prefix` | Folder-mode filename prefix, default empty. |
| `--mask-path` | Optional 2D TIFF; nonzero pixels limit scheduled output. |
| `--resolution` | Zarr pyramid key, default `0`; bare arrays require level 0. |
| `--num-workers`, `--workers` | DataLoader workers, default `4`. |
| `--prefetch-factor` | Per-worker prefetch factor, default `2`. |
| `--overlap` | Inference overlap fraction in `[0,1)`, default `0.25`. |
| `--stride` | Explicit pixel stride; overrides `--overlap`. |
| `--blend-mode` | `auto`, `constant`, `gaussian`, or `hann`; default `auto`. |
| `--layer-start`, `--layer-end` | Half-open source-Z selection before centered depth cropping; negative indices count from the end. |
| `--batch-size` | Batch size per selected device, default `1`. |
| `--direction` | `forward`, `reverse`, or `both`; default `forward`. |
| `--amp-dtype` | `auto`, `default`, `fp16`, or `bf16`; `auto` reads checkpoint precision. |
| `--tta-mirror` | Average valid mirror variants. |
| `--tta-batch-size` | Maximum mirror variants evaluated together. |
| `--gpus` | Unique comma-separated CUDA IDs, for example `0,1`; omit for automatic CUDA/CPU selection. |
| `--compile-mode` | `torch.compile` mode, default `reduce-overhead`. |
| `--no-compile` | Use eager inference. Multiple selected GPUs also disable compilation. |

Flat inference requires a checkpoint with `mode: "flat"` and a square Y/X
crop. It prefers `ema_model`, then common flat state-dict aliases, then
`model`. The output is a tiled LZW uint8 BigTIFF. Probabilities are clipped to
`[0,1]`, multiplied by 255, and truncated to uint8.

For checkpoint compatibility, flat inference has a narrower preprocessing
adapter than training: `divide` requires `divisor: 255`, and `clip_divide`
requires `clip_min: 0`, `clip_max: 200`, and `divisor: 255`. Other configured
normalization names follow the robust flat-inference path.

## Native 3D inference

A tifxyz directory must contain `x.tif`, `y.tif`, `z.tif`, and
`volume_source.txt`. The latter contains a volume URL, absolute path, or a path
relative to the tifxyz directory.

Inspect the occupied-chunk plan without creating output:

```bash
uv run --extra models python -m vesuvius.ink_detection.inference.infer_full3d_tifxyz \
  /data/ink_9um/labels/0139/native_9p362_level0/w035 \
  /data/ink_9um/checkpoints/native-best.pth \
  /data/predictions/w035.ome.zarr \
  --plan-only
```

Run inference with eight mirror variants and a bounded volume cache:

```bash
uv run --extra models python -m vesuvius.ink_detection.inference.infer_full3d_tifxyz \
  /data/ink_9um/labels/0139/native_9p362_level0/w035 \
  /data/ink_9um/checkpoints/native-best.pth \
  /data/predictions/w035.ome.zarr \
  --tta \
  --cache-dir /data/cache/ink-native \
  --cache-max-gb 100 \
  --gpus 0
```

| Argument | Meaning |
|---|---|
| `tifxyz_dir checkpoint output_zarr` | Required geometry, config-bearing checkpoint, and output path. |
| `--resolution` | Scroll-volume pyramid level, default `0`; tifxyz coordinates are scaled accordingly. |
| `--overwrite` | Replace an existing output; otherwise existing output is refused. |
| `--batch-size` | Batch size per selected device, default `1`. |
| `--num-workers`, `--workers` | DataLoader workers, default `4`. |
| `--prefetch-factor` | Per-worker prefetch factor, default `2`. |
| `--downsample-workers` | Pyramid downsampling threads, default `1`. |
| `--overlap` | Native patch overlap fraction, default `0.5`. |
| `--chunk-halo` | Occupied-chunk expansion radius, default `1`. |
| `--write-region` | Write `expanded` chunks by default, or only `occupied` chunks. |
| `--blend-mode` | `gaussian` by default, or `constant`. |
| `--tta` | Average all eight Z/Y/X mirror variants. |
| `--tta-batch-size` | Maximum variants evaluated together. |
| `--amp-dtype` | `auto`, `default`, `fp16`, or `bf16`; `auto` reads checkpoint precision. |
| `--compile-mode` | `torch.compile` mode, default `reduce-overhead`. |
| `--no-compile` | Use eager inference. Multiple selected GPUs also disable compilation. |
| `--gpus` | Unique comma-separated CUDA IDs; omit for automatic CUDA/CPU selection. |
| `--plan-only` | Print volume/chunk/patch counts without building or running the model or creating output. The checkpoint is still deserialized for its config. |
| `--max-target-chunks` | Refuse plans larger than this positive count. |
| `--cache-dir`, `--cache-max-gb` | Optional Zarr-3 compressed-volume cache root and per-volume decimal-GB budget. |
| `--log-level` | Python logging level, default `INFO`. |

Native inference requires a checkpoint whose mode is `full_3d` or
`full_3d_single_wrap`; it prefers `ema_model` and otherwise loads `model`.
`--plan-only` still reads the checkpoint config, tifxyz coordinates, and input
volume because those determine the plan. Level 0 is a sparse uint8 Zarr-v2
array in scroll Z/Y/X coordinates. Five rounded-mean levels are derived from
it, producing a six-level OME-Zarr pyramid with ceil-halved shapes.

## Labeling loop

1. Train from Zarr labels and a surface volume.
2. Run flat inference and inspect the uint8 TIFF probabilities.
3. Extend `<segment>_inklabels.tif` and
   `<segment>_supervision_mask.tif`; add a validation mask if needed.
4. Run `create_label_zarrs` on the scroll-first family directory.
5. Retrain, optionally setting `label_version` when a coordinated version is
   required.

Keep supervision explicit: ink labels mark positive pixels, while the
supervision mask defines which pixels participate in loss and metrics.

## Tests

Run the self-contained ink-detection coverage:

```bash
uv run --no-sync pytest tests/ink_detection -q
```

Run the package suite without slow or network tests:

```bash
uv run --no-sync pytest tests -m "not slow and not network"
```

## Limitations

- Supported dataset modes are only `flat`, `full_3d`, and
  `full_3d_single_wrap`.
- ResNet3D, TimeSformer/container inference, `normal_pooled_3d`, mean-teacher,
  Betti matching, sweeps, and the embedded sample viewer are not available in
  this package.
- Positive native label or supervision dilation enters the CUDA/cuCIM path;
  use zero dilation where that capability is unavailable.
- Flat TIFF and native OME-Zarr encoding intentionally differ: flat output
  truncates scaled probabilities, while native output rounds probabilities and
  derived pyramid means.
- `torch.compile` is best effort. If compilation fails, inference logs a
  warning and continues eagerly.
