# Dinovol

Dinovol is an implementation of DINO-style self-supervised pretraining on 3D
volumes.

The canonical source is
[`ScrollPrize/villa/dinovol`](https://github.com/ScrollPrize/villa/tree/main/dinovol).
Run project commands from the monorepository's `dinovol/` directory.

- `dinov2_eva` is adapted from [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/architectures/dinov2_eva.py), with minimal changes.
- The augmentation library is a modified [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2).
- Normalization is mostly borrowed from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
- RoPE is adapted from the [DINOv3 implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py) and extended to support 3D.

This implementation is still incomplete. Pretraining works, but finetuning has
not yet been implemented.

Use the `v3` preset for new runs, as selected by every example recipe. It uses
plain SwiGLU with a 2x hidden width and no query/key/value projection biases,
axial 3D RoPE, separate local/global CLS output normalization, constant LR after
warmup, constant weight decay and teacher momentum, and KoLeo groups of up to 16.
The `v2` preset and configs without `model_type` retain their older defaults for
checkpoint compatibility. Explicit config values override preset defaults.
To select the new defaults, set `model.model_type` to `v3`:

```json
{
  "model": {
    "model_type": "v3",
    "embedding_type": "default",
    "global_crops_size": [96, 96, 96],
    "local_crops_size": [48, 48, 48]
  }
}
```

## Pretraining on 8–9 µm scroll volumes

[`dinovol_2/configs/pretrain_8_9um.json`](dinovol_2/configs/pretrain_8_9um.json)
is a standalone base-pretraining config with 128³ global crops, 64³ local crops,
and `volume_scale: 0` for every volume. It uses a CNN feeding a width-288 `v3`
backbone with 10 full-attention transformer blocks, a 2,048-wide SwiGLU FFN, and six attention
heads, batch size 16 per GPU without gradient checkpointing, two global views,
and two local views. Gram and
point supervision are disabled for this initial stage.
The FFN uses `mlp_ratio: 7.111111111111111` (2,048 / 288); its packed input
projection produces two 2,048-wide branches before gating.

Compilation is enabled for this recipe with the top-level setting:

```json
"compile": {"enabled": true, "backend": "inductor", "mode": "default", "dynamic": {"backbone": false}}
```

The trainer compiles the student and teacher backbones (CNN plus transformer)
and DINO/iBOT projection heads in place using `nn.Module.compile`. A Gram teacher
backbone is also compiled if present. Losses, token selection, optimizer/EMA
updates, and data loading remain eager. Parameter names and optimizer references
are unchanged, so checkpoints can be resumed with compilation enabled or disabled.
Compilation does not add CNN checkpointing or change precision settings.
Compiled backward graphs are captured with autocast disabled for backward,
matching the trainer's existing AMP forward / non-autocast backward sequence.

Other configs default to compilation disabled; `"compile": false` disables it.
Optional settings are `backend`, `mode`, `dynamic` (default `null`, automatic
dynamic-shape handling), and `fullgraph` (default `false`, allowing graph breaks).
`dynamic` may also map `backbone`, `dino_head`, and `ibot_head` to `true`,
`false`, or `null`; unlisted modules use `null`. This recipe compiles backbones
statically, giving one graph per global/local crop shape instead of a slower
dynamic-shape recompile, while the iBOT head keeps automatic handling for its
varying number of masked tokens. With the inductor backend the trainer disables
Inductor's mix-order reduction, which in torch 2.10 returns the mask-token
gradient with the wrong shape in channels-last CNN backward graphs (8+ masked
128³ crops); disabling it had no measurable throughput cost.
Initial calls compile lazily; global/local shapes and training/evaluation paths
can each require warmup. Compilation failures are not silently suppressed.
The default mode avoids explicitly enabling CUDA-graph modes that reserve extra
workspace memory. Measure throughput and peak VRAM after warmup on the target GPU;
compiled execution can change floating-point rounding and random-number streams.
For compiler diagnostics, run with `TORCH_LOGS=graph_breaks,recompiles`.

The recipe also enables these throughput and memory settings. Each is off unless
a config sets it:

- `ibot_fused_last_layer` (top level): the heads return the iBOT bottleneck
  features and last-layer weight, and the loss forms the
  `[n_masked, ibot_out_dim]` logits, Sinkhorn-Knopp targets, and gradients one
  chunk of `ibot_masked_loss_chunk_size` rows at a time. Sinkhorn-Knopp reduces to
  per-prototype scales `b` with targets `softmax(teacher_logits / T + log b)`, so
  three passes over recomputed teacher logits replace the stored matrices. The
  math is unchanged; only floating-point rounding differs.
- `model.deeper_embed_channels_last`: CNN activations and weights use the
  channels-last 3D layout, which cuDNN's tensor-core convolutions use natively.
  Weights are converted after initialization; checkpoints are layout-independent.
- `model.deeper_embed_batch_chunk_size: null`: the CNN embeds the whole batch at
  once instead of one sample at a time (the default of 1).
- `fused_optimizer`: fused CUDA AdamW.
- `cuda_expandable_segments`: the CUDA caching allocator uses expandable
  segments, avoiding fragmentation-driven cache flushes near full memory.
- `dataset.occupancy_level`: the pyramid level (5 here, 32× coarser) whose
  nonzero voxels, dilated by one cell, bound each source window's nonzero
  fraction. Windows that cannot reach `nonzero_threshold` are rejected before any
  read, so the accepted windows and their distribution are unchanged. Masked-out
  chunks are absent from the S3 stores, so these reads were otherwise wasted.
  The coarse masks are read once at startup and only cover the usable bounding box.
- `dataset.read_ahead`: each loader process reads this many future source
  windows in background threads while it augments the current sample. Window
  locations are still drawn in the loader process, so sampling stays
  deterministic, but enabling it changes the order of random draws.
- `dataset.read_retries` (default 5) and `dataset.read_retry_backoff` (default
  1 s): source-window and occupancy reads retry transient network errors
  (dropped connections, truncated responses, timeouts) with doubling backoff,
  re-reading the same window, so a single dropped S3 connection no longer kills
  a run. Each retry is printed to stderr.
- `dataset.gaussian_blur: false` and `dataset.global_crop_scale: [0.18, 0.45]` /
  `dataset.local_crop_scale: [0.023, 0.057]`: views are resampled only 0.87–1.18×
  from the 192³ source window (global 108–147 → 128 voxels, local 55–74 → 64)
  and are never blurred, so one-voxel structures survive augmentation. The zoom
  range still covers the 7.9–9.4 µm spread in voxel size between scans.
- `num_workers: 24`, `prefetch_factor: 2`, `persistent_workers: true`: with
  read-ahead, one worker produces about 1.8 samples/s (augmentation-bound), so 24
  workers comfortably exceed the GPU's ~20 samples/s on an H100.

The teacher forward now runs before the student forward, so its no-grad
intermediates are freed before student activations accumulate.

The model option `attention_type` accepts `"full"` (the default, used in this recipe)
or `"axial"`. Each axial block has three sequential attention
sublayers along D, H, and W, each with its own normalization, projections, and
residual update, followed by one FFN. Patch queries attend to their spatial
line plus CLS/register tokens; CLS/register queries attend to all tokens on
each pass. Thus global crop representations remain available for DINO, and
spatial outputs remain available for iBOT. RoPE is applied using the original
grid coordinates before grouping lines. `rope_type` independently selects the
positional encoding and does not select the attention pattern.

Axial attention supports variable crop grids and gradient checkpointing. It
reduces attention interactions but adds two sets of attention projections per
3D block; runtime and training quality should be measured on the target GPU.
Full-attention checkpoints keep their existing parameter layout when
`attention_type` is omitted. Switching to axial adds parameters and requires
a matching checkpoint or a new training run.

The CNN uses the existing `deeper` embedding with a 16-channel full-resolution
stem: one 3×3×3 stride-1 convolution, normalization, and activation. Four residual
stages follow, with 16, 48, 128, and 192 channels, strides 1, 2, 2, and 2, and
1, 2, 2, and 1 basic residual blocks (two 3×3×3 convolutions per block).
The first stage operates entirely at full resolution; subsequent stages
downsample in the first convolution of their first block. This gives three
convolutions before the first downsampling, enough to resolve one-voxel
structure while keeping full-resolution activations small. Capacity sits in the
64³ and 32³ stages; a single 192-channel block at token resolution is followed
by a 1×1×1 projection to the 288-wide EVA tokens. Residual skip connections
retain their channel projections. Total stride is 8, giving 4,096 global
tokens and 512 local tokens, before CLS/register tokens. The CNN has 3.5M
parameters and the backbone 24.5M; the CNN trains at the full learning rate
(`patch_embed_lr_mult: 1.0`).

The recipe uses a 192³ source sampling window, from which augmented crops are
drawn. CNN convolutions use zero padding, and the model receives exactly the
128³ global and 64³ local crops. There is no halo expansion, extra context read,
or cropping of the embedded grid. The output directory is
`dinovol_pretrain_8_9um_cnn192proj_10blocks_ffn2048_full_g128_p8_stem16/` to avoid automatically resuming
incompatible checkpoints from earlier architectures.

The configurable CNN options under `model` are:

```json
{
  "embedding_type": "deeper",
  "embed_dim": 288,
  "num_heads": 6,
  "patch_size": [8, 8, 8],
  "deeper_embed_stage_channels": [16, 48, 128, 192],
  "deeper_embed_stage_strides": [1, 2, 2, 2],
  "deeper_embed_depth_per_level": [1, 2, 2, 1],
  "deeper_embed_use_stem": true,
  "deeper_embed_stem_channels": 16,
  "deeper_embed_stem_type": "conv",
  "deeper_embed_final_projection": true
}
```

Without a final projection, the last stage must have `embed_dim` channels.
Stage channel/depth counts must match the stride list, whose per-axis product
must equal `patch_size`. Each stride entry can be a scalar or a per-axis list
of 1s and 2s. When omitted, strides are derived from `patch_size` as before.
Stem width defaults to 32, and stem type defaults to the embedding block style
(a residual stem for the standard deeper embedding); `"conv"` selects a single
convolution with normalization and activation.
Omitted CNN architecture options preserve the existing deeper embedding's
parameter layout. All CNN embeddings now use zero padding; older checkpoints
can load with matching architecture, but their boundary behavior changes.
Spatial CNN chunking and the separate `*_view_size` dataset settings have been
removed; use `*_crop_size` and optionally `deeper_embed_batch_chunk_size`.
The CNN regression tests check the recipe's architecture
on a 32³ synthetic crop, gradients, masking, and checkpoint reconstruction;
the pretraining smoke suite also exercises the projection-free path with a
smaller CNN and synthetic volume.

The volume list is pinned from the public
[scan metadata](https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json),
retrieved on 2026-10-10. It contains one masked intensity OME-Zarr per eligible
scroll sample: 27 ESRF volumes at 8.640–9.366 µm, plus four Diamond volumes at
7.910 µm (treated as approximately 8 µm). ESRF is preferred within that resolution
range; higher-resolution ESRF scans are not substituted at another pyramid level.
`sample_id`, `pixel_size_um`, and `scan_location` in each entry document the
selection; the loader uses `volume_path` and `volume_scale`. Sampling retains the
existing weighting by valid crop starts, rather than equal weight per scroll.

The Diamond fallbacks are PHerc0172, PHerc0332, PHerc1667, and PHercParis4.
PHerc1299, PHercMAN5, PHercMANB, and PHercParis3 are excluded because the catalog
has no intensity volume near the requested resolution at scale 0. For duplicate
candidates, PHerc0139 uses `20250728140407` (the standard reconstruction; the
pag0/pag50 alternatives have documented masking problems), PHerc0172 uses the
newer cropped export `20241024131839`, and PHercParis4 uses `20230205180739`
for its larger axial coverage.

Run from `dinovol/` with the existing environment:

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python -m dinovol_2.pretrain dinovol_2/configs/pretrain_8_9um.json
```

The URLs are public and require no credentials file. Volume selection and
scale-0 array metadata were checked; GPU memory requirements and training quality
have not been measured for this recipe. Adjust `batch_size` for the available GPU.

## License

Original Dinovol code is available under the [MIT License](LICENSE).
Third-party code retains its upstream terms; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Backbone and learning-rate options

The following settings can be added to a training config:

```json
{
  "lr_schedule": "constant",
  "warmup_steps": 1000,
  "koleo_distributed": true,
  "koleo_loss_group_size": 16,
  "model": {
    "model_type": "v3",
    "rope_type": "axial",
    "scale_mlp": false,
    "mlp_ratio": 2.0,
    "qkv_bias": false,
    "untie_cls_and_patch_norms": false,
    "untie_global_and_local_cls_norm": true
  }
}
```

- `lr_schedule` defaults to `constant` for `v3`, and `cosine` for older presets. `constant` uses the existing linear
  warmup and then holds the configured `lr`, including beyond `max_iterations`;
  it ignores `min_lr`. Last-layer freezing and parameter-group LR multipliers
  still apply. Weight decay, teacher momentum, and temperature retain their
  own schedules. In `v3`, weight decay defaults to a constant `0.04` and teacher
  momentum to a constant `0.994`; overriding their initial values also changes
  their default final values. Set `weight_decay_end` equal to `weight_decay` and
  `final_momentum_teacher` equal to `momentum_teacher` to hold those constant too.
  `warmup_steps: 0` disables LR warmup; otherwise the default is 10% of the run.
- `scale_mlp`, `mlp_ratio`, and `qkv_bias` default to `false`, `2.0`, and
  `false` in `v3`, following the DINOv3 7B pretraining block. `scale_mlp: false`
  removes the extra normalization inside SwiGLU; the usual pre-attention and
  pre-MLP norms remain. Our ratio specifies the effective hidden width directly:
  DINOv3's `ffn_ratio: 3` applies an internal two-thirds adjustment. Older presets
  retain `true`, `8/3`, and `true`. To load checkpoints from the earlier `v3`
  architecture, preserve their saved model configuration or explicitly set all
  three old values (`mlp_ratio: 2.6666666666666665` in JSON); changing these
  settings changes parameter shapes and state-dict keys.
- `rope_type` selects the existing 3D `axial` or learnable per-block `mixed`
  implementation. The default is `axial` for `v3`, `mixed` for `v2`, and `axial`
  for the original unversioned model. Both support variable crop sizes and coordinate rescaling.
  The attention head dimension must be divisible by six for 3D RoPE.
- `untie_cls_and_patch_norms` gives CLS/register tokens their own output norm;
  patch tokens keep the original `norm`.
- `untie_global_and_local_cls_norm` adds a separate CLS/register output norm for
  local crops during student training. Evaluation and the EMA teacher use the
  global norm. This flag defaults to `true` for `v3` and `false` for older presets;
  `untie_cls_and_patch_norms` remains `false` in all presets.
- `koleo_distributed` defaults to `true`: with DDP, KoLeo compares each view's
  embeddings across ranks, with gradients through gathered embeddings. This
  expects equal per-rank batch sizes, as supplied by the training loaders.
  Set it to `false` for local-only comparisons. A comparison batch with fewer
  than two samples produces zero loss and gradient.
- `koleo_loss_group_size` defaults to `16` for `v3`, and `null` (the whole
  comparison batch) for older presets. Groups are contiguous in rank-major order,
  independently for each global view. If fewer than 16 samples are available,
  all available samples are used. Larger comparison batches must be divisible
  by the requested group size; choose a divisor or `null` otherwise. Grouping
  limits neighbour selection; the current implementation still gathers all ranks.

Model head dimensions and `pretrained_weights` are preserved in the resolved
training config. When enabling separate norms on an older checkpoint, missing
norm weights are initialized from its trained shared norm (or its existing CLS
norm for the new local norm). Use `model.pretrained_weights` to start a new run
when changing normalization options; resuming optimizer state requires the same
parameter layout. Keep `rope_type` consistent with the checkpoint: switching
between learned mixed and axial RoPE is an architecture experiment, not an exact
checkpoint conversion.

## Masking

Masked student views always zero the selected voxel regions before patch
embedding, then replace the selected embeddings with the learned mask token.
Each loss-mask position covers one `patch_size` region on the output grid.
Masking happens on the prepared model input, before convolution. The original
crop remains unchanged for the unmasked teacher.

For `embedding_type: "default"`, this is mathematically equivalent to the former
token-only masking, including parameter gradients. For `"deeper"`, it prevents
neighboring embeddings from observing the hidden voxel values and changes
training behavior, including when resuming an older checkpoint. There is no
legacy masking mode or configuration switch; checkpoint parameter shapes stay
unchanged. Output/gradient equivalence and hidden-voxel isolation are covered by
`tests/test_voxel_masking.py` in the standard test suite.

## Gram Anchoring And HR Adaptation

`pretrain.py` now supports the three-stage DINOv3-style workflow:

- base pretraining with DINO + iBOT + KoLeo
- late dense-feature refinement with Gram anchoring
- short mixed-resolution HR adaptation with Gram anchoring kept on

The new config surface is:

- top-level `gram`
  - `enabled`
  - `loss_weight`
  - `teacher_checkpoint`
  - `teacher_refresh_every`
  - `teacher_refresh_start_step`
  - `normalized`
  - `img_level`
  - `remove_neg`
  - `remove_only_teacher_neg`
- dataset keys
  - `gram_teacher_crop_size`
  - `gram_teacher_no_augmentations`
  - `variants`

When `gram.enabled=true`, the trainer builds a frozen Gram teacher backbone, loads it from `gram.teacher_checkpoint` when provided, refreshes it from the live EMA teacher on the configured cadence, and adds an image-level Gram loss on patch features. Gram-teacher crops are paired with each global crop from the exact same sampled 3D region. With `gram_teacher_no_augmentations=true`, they receive normalization and the same geometric transforms as the student, but no intensity/noise augmentation. Shared rotations, flips, and transpositions preserve patch correspondence even at different crop resolutions.

Mixed-resolution HR adaptation is enabled by adding `dataset.variants`, where each variant defines its own crop sizes and sampling ratio. The trainer builds one dataloader per variant and samples them with the configured weights. All embedding types consume the configured crop sizes directly.

`dinovol_2/example_config.json` remains runnable as a base-pretraining config and also includes a `recipes` object with three complete examples:

- `recipes.base_pretrain`
- `recipes.gram_refinement`
- `recipes.hr_adaptation`

The important stage-specific overrides are:

```json
{
  "gram": {
    "enabled": true,
    "loss_weight": 2.0,
    "teacher_checkpoint": "/path/to/previous/checkpoint.pt",
    "teacher_refresh_every": 10000
  },
  "model": {
    "pretrained_weights": "/path/to/previous/checkpoint.pt",
    "pretrained_backbone_only": false
  }
}
```

For HR adaptation, add weighted crop variants:

```json
{
  "dataset": {
    "variants": [
      {
        "ratio": 0.3,
        "global_crop_size": [128, 128, 128],
        "local_crop_size": [48, 48, 48],
        "gram_teacher_crop_size": [160, 160, 160]
      },
      {
        "ratio": 0.7,
        "global_crop_size": [160, 160, 160],
        "local_crop_size": [80, 80, 80],
        "gram_teacher_crop_size": [192, 192, 192]
      }
    ]
  }
}
```

To sanity-check one batch end to end:

```bash
uv run python -m dinovol_2.verify dinovol_2/example_config.json --no-amp
```

To run the synthetic smoke suite:

```bash
uv run python -m unittest discover -s tests -v
```

With an existing environment, run without installing or updating dependencies:

```bash
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

The suite includes synthetic volume smoke tests and regressions for checkpoint
initialization, output norms, both RoPE implementations, constant LR, paired Gram
geometry, and two-rank KoLeo gradients. It does not establish downstream quality;
validate training changes on real held-out volumes before a long run.

## Point-supervised pretraining

Pretraining can optionally reinforce patch embeddings with sparse annotated points while DINO, iBOT, KoLeo, and Gram remain active. The top-level defaults are:

```json
{
  "point_supervision": {
    "enabled": true,
    "sampling_probability": 0.1,
    "loss_weight": 0.05,
    "different_type_margin": 0.0,
    "max_points_per_view": 64
  }
}
```

Attach annotations to each applicable volume (paths are intentionally omitted from the runnable example config):

```json
{
  "volume_path": "/path/to/volume.zarr",
  "volume_scale": 1,
  "point_collections": [
    {"path": "~/points/fibers.json", "type": "fiber"},
    {"path": "annotations/surface.json", "type": "surface"}
  ]
}
```

Each annotation file must be version 1 and contain a `collections` array whose entries contain `points` arrays with `p: [x, y, z]`. The configured `type` applies to every point in that file; equal type strings have equal meaning across volumes. `~` is expanded, while other relative paths remain relative to the process working directory.

Coordinates are scale-0 voxel centers in XYZ order. Dinovol converts them to ZYX and maps voxel centers to the selected pyramid level using the co-registered level-shape ratio. Point-centered sampling and the point loss are inactive when the feature is disabled or no volume has `point_collections`. Only training samples and the two global views are supervised; validation, monitor batches, and local views keep their existing sampling behavior.

## Optional Task Eval During Pretraining

`pretrain.py` can optionally run small downstream segmentation trainings during pretraining.

- set `task_eval_every` to a positive step cadence to enable it
- choose `eval_task` as `both`, `surfaces`, or `ink`
- set `eval_task_train_iters` to control the mini-training length, default `500`
- set `eval_task_decoder_type` to `simple` or `patch_encode_decode`

The task data is downloaded with `python -m dinovol_2.eval.download_data --task both`.

- `both` now means `surfaces` plus `ink`
- `surfaces` is resized 2x before crops are drawn
- `surfaces` and `ink` each use the first 10 sorted samples as the deterministic validation set
- `ink` is not resized before crops are drawn
- train and validation crops are taken from precomputed chunks that contain some foreground and at least 50% background in supervised voxels
- the saved validation image contains one row per validation sample, with image / label / prediction panels
- for `ink`, voxels with `supervision_mask == 0` are ignored and supervised unlabeled voxels are treated as background
- for `ink`, loss/metrics and saved previews use a max projection across Z to match the flat ink trainer

## Napari visualizer

There is a small napari helper for checkpoint inspection at `dinovol_2/eval/napari_visualizer.py`.

Run it with:

```bash
python -m dinovol_2.eval.napari_visualizer
```

Workflow:

- open an OME-Zarr from the widget, click `Load Scales`, choose the desired scale, and click `Open Zarr`
- draw a rectangle in the generated `*_bbox` shapes layer; this 2D YX bbox is applied across the full Z span of the selected scale
- add one or more points in a `Points` layer
- choose a `pretrain.py` checkpoint, image layer, and points layer in the dock widget
- click `Cache Embeddings`
- click `Show Feature PCA` to render a 3-channel PCA view of the cached patch embeddings
- optionally enable `Otsu Foreground Mask` and set `Mask Dilation` before creating the PCA layer
- click `Similarity For Selected Points` or `Similarity For All Points`

The widget rebuilds the teacher backbone from the saved checkpoint config, computes a patch embedding grid only inside the active bbox for the selected OME-Zarr scale, and limits the PCA and cosine-similarity outputs to that same crop. The dock widget opens on the bottom of the napari window.
