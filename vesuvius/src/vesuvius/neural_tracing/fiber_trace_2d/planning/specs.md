# 2D Fiber Trace Initial Loader Specs

## 3D CP-Centered Fiber Model Variant

- The 3D CP model lives in a sibling package,
  `vesuvius.neural_tracing.fiber_trace_3d`. It must not replace or reinterpret
  `fiber_trace_2d` configs, strip geometry, Trace2CP tooling, or 2D training.
- 3D training samples ordinary CP-centered ZYX volume blocks from the selected
  base-volume Zarr level. It must not build fiber-aligned 3D strips or slices
  for input loading.
- Dataset entries use the same Lasagna-manifest-first convention as 2D:
  `base_volume_path`, `base_volume_scale`, required
  `lasagna_manifest_path`, and `fiber_paths`/`fiber_glob`.
- JSON and NML fiber parsing, control-point exactness, and optional
  dataset-level XYZ affine transforms follow the existing
  `fiber_trace_2d.fiber_json` semantics.
- `base_volume_scale` selects both the Zarr level read by the 3D loader and the
  voxel scale at which CP-centered patches are sampled.
- The 3D sample stream is deterministic pseudo-random by configured `seed` and
  covers every configured control point once per pass before repeating. Changing
  batch size or step count may truncate/extend the consumed prefix, but must not
  reshuffle earlier samples.
- 3D training uses the same raw-stream/data-stream split as 2D:
  `training.max_sample_index` limits CP/data sample selection only, while
  geometric and value augmentation parameters are seeded by the unbounded raw
  training sample index. Reusing a bounded CP/data prefix must not replay the
  same augmentation transforms on each repeat.
- Public 3D loader calls use `sample_index` as that raw/global deterministic
  stream index. There is no separate public augmentation-index mode:
  `sample_index_limit` is the only mechanism that changes bounded CP/data
  selection, and augmentation seeding remains tied to the raw `sample_index`.
- With `training.max_steps = 0`, 3D training repeats the deterministic training
  stream indefinitely until interrupted. Positive `max_steps` values are
  absolute target training steps, including for resumed runs.
- 3D geometric augmentation is represented by explicit coordinate maps before
  the final volume patch is materialized. `backward_source_zyx` maps output
  voxels to selected-level source-volume coordinates for image sampling.
  Source fiber points are mapped to output-patch coordinates with the matching
  analytic forward transform built from the same augmentation parameters. The
  image, transformed fiber line, transformed CP, and direction targets must
  therefore see the same geometry without dense inverse search.
- V0 3D geometric augmentations support CP-local shift, isotropic scale,
  arbitrary 3D rotation, and independent axis flips. Non-zero 3D shear/skew and
  ringing artifact keys are rejected until their semantics are specified.
- 3D smooth displacement is opt-in through
  `augment_smooth_displacement_mode` (`none`, `1d`, `2d`, `3d`),
  `augment_smooth_displacement_amplitude_zyx`,
  `augment_smooth_displacement_control_spacing_zyx`, and
  `augment_smooth_displacement_probability`. Smooth modes must use explicit
  paired map construction, matching the 2D fused-map contract. Runtime paths
  must not invert one map direction into the other by search, brute-force
  nearest lookup, iterative solving, or formula re-evaluation. The current 3D
  mode uses explicitly invertible 1D/2D offsets and 3D triangular coupling
  stages.
- The 3D loader must sample the final regular 3D patch through explicit
  coordinates using the VC3D blocking coordinate sampler. It must not load an
  oversized axis-aligned zarr crop and then resample that crop with torch
  `grid_sample` for normal training. Array-backed tests may use the
  NumPy/trilinear fallback sampler. Value-only augmentations happen after
  sampling as torch tensor operations.
- VC3D blocking coordinate sampling means strict requested-level sampling:
  every required requested-level chunk is fetched/decoded and locally pinned
  before sampling starts, scale fallback is disabled, and returned stats report
  `requested_level_only: true`, `fallback_levels: 0`, and `missing_chunks` for
  genuinely absent requested-level chunks. Only those truly missing
  requested-level chunks may render black. Chunk I/O/decode errors must fail
  loudly. The returned sampler `valid_mask` is only geometry/sample coverage;
  it must not be treated as proof that requested-level data was used.
- V0 3D value augmentations support normalization, brightness, contrast, gamma,
  noise, and separable isotropic Gaussian blur. Opt-in anisotropic blur is
  configured with `augment_anisotropic_blur_probability`,
  `augment_anisotropic_blur_sigma_along`,
  `augment_anisotropic_blur_sigma_across`,
  `augment_anisotropic_blur_orientation`, and
  `augment_anisotropic_blur_roll_degrees`. It is a value augmentation after
  coordinate sampling, not a geometric transform.
- The 3D model output layout is grouped by direction branch. Each branch has
  seven channels: six Lasagna 3x2 ambiguous direction channels followed by one
  sigmoid sheet/fiber-presence channel. Legacy/single-branch configs use one
  group (`7` channels); active multi-direction training configs use
  `model_3d.direction_branch_count: 2` and `14` output channels.
- Branch 0 preserves the legacy channel positions: channels `0:6` are
  direction and channel `6` is presence. Branch 1 uses channels `7:13` and
  channel `13`.
- Each branch's six direction channels use Lasagna's double-angle projection layout:
  `dir0_z,dir1_z` for `(tx,ty)`, `dir0_y,dir1_y` for `(tx,tz)`, and
  `dir0_x,dir1_x` for `(ty,tz)`.
- Direction supervision is computed from the transformed 3D line tangent and is
  masked to positive fiber-neighborhood voxels. Projection-magnitude weighting
  may downweight channels whose projection is nearly degenerate.
- Positive direction/presence supervision chooses exactly one branch per sparse
  positive supervision point. The chosen branch is
  `argmax(abs(dot(decoded_predicted_axis, target_axis)) * predicted_presence)`,
  using a detached score for the discrete routing decision. Direction loss and
  positive presence BCE apply only to the selected branch at points included by
  `presence_mask`. The unselected positive branch is not trained as negative at
  that positive point.
- During two-branch 3D training, positive routing includes a batch-local
  anti-collapse repair over 4-voxel spatial groups within each patch. Training first
  averages detached branch choice scores per `(patch, 4x4x4 chunk)`. If either
  branch receives fewer than 10% of grouped positive supervision, the
  underrepresented branch takes the missing quota from groups currently assigned
  to the other branch, sorted by the underrepresented branch's grouped detached
  choice score. The chosen group branch is broadcast back to all sparse positive
  points in that group. If both branches already meet the floor, routing is
  unchanged. Test/eval metrics do not apply this repair; they use the raw
  detached argmax routing.
- Negative presence supervision remains global: all branches are supervised as
  negative where the dense presence target is negative inside the valid/reachable
  patch interior. For CP-only samples, edge voxels that could not contain a CP
  because of shift augmentation are ignored for presence loss. For NML
  dense-line samples, presence supervision uses the full valid patch because the
  centerline can provide positives and negatives in that region.
- Presence loss is normalized by patch and branch before aggregation. Sparse
  positive presence BCE is averaged per selected `(patch, branch)` group; dense
  negative presence BCE is averaged per `(patch, branch)` group. The available
  positive and negative group-normalized terms are then summed directly, without
  an additional global positive/negative balancing factor.
- 3D training defaults `training.direction_weight` to `10.0` and
  `training.presence_weight` to `1.0`, so direction loss is 10x stronger than
  presence loss unless the config overrides it.
- 3D target generation is source-format dependent. NML fibers use dense
  supervision along all fiber-line segments that overlap the patch. The
  transformed output-space segments are clipped/rasterized directly into the
  patch target volume; they must not be generated by a full voxel-by-segment
  nearest search or by inverting sampled image coordinates. Non-NML fiber
  sources supervise only the sampled CP neighborhood for direction and
  presence.
- The first 3D model uses branch-routed direction plus presence losses.
  Contrastive embedding remains unsupported by default in the 3D V0 path.
- The 3D fiber model defaults to `BatchNorm3d`; configured `batch_size` is the
  actual BatchNorm batch because the trainer has no internal micro-batching.
  `model_3d.normalization: "none"` remains supported for explicit ablations.
- `batch_size` is the actual CP-patch batch passed through the 3D model in one
  forward/backward call. The 3D trainer does not support internal
  micro-batching; any BatchNorm statistics must come from the real configured
  batch.
- The S1A NML 3D training config uses `patch_shape_zyx: [192,192,192]`,
  `augment_shift_zyx: [48,48,48]`, and a fixed six-stage U-Net depth
  (`[16,32,64,128,256,512]`) so the deepest feature map remains appropriate
  for 192-voxel patches.
- `train_s1a_nml_all_64_sd2.json` is a fast experimental S1A NML config for
  64-voxel patches at `base_volume_scale: 2`. It keeps the same implemented
  augmentation families enabled at smaller magnitudes appropriate for that
  patch size: affine shift/rotation/scale/flip, value brightness/contrast/
  gamma/noise, isotropic blur, smooth displacement, and anisotropic blur.
  Shear/skew and ringing remain unsupported and must not appear as enabled
  keys in this config.
- 3D training TensorBoard visualization logs CP-centered slice sheets at
  `training.sample_vis_interval`. By default, up to four batch samples are
  shown; `training.sample_vis_count` / `train_sample_vis_count` and
  `training.test_sample_vis_count` control the side-by-side train/test sample
  counts. Each sample block has five rows: the `yx`, `zx`, and `zy` principal
  planes, a longitudinal slice containing the GT CP tangent, and a
  perpendicular/cross slice whose plane normal is the GT CP tangent. Each row
  has seven columns: volume image with projected GT line and model-predicted/
  fitted CP direction overlay where applicable, target/context presence,
  branch presence for the output whose decoded direction is closer to the slice
  normal by `abs(dot(axis, normal))`, the other branch presence, max branch
  presence, min branch presence, and average branch presence. The target/context
  presence panel must visualize the carried transformed fiber-line segment
  metadata even for JSON/non-NML CP-only samples where loss supervision remains
  CP-only. The two oblique rows must project/rasterize transformed line
  segments into their oblique slice frame for both image overlay and
  target/context presence. Dense-line/NML samples must carry the transformed CP
  tangent so the GT-tangent and perpendicular rows are constructed from the
  actual local target tangent. The GT line overlay includes target-line
  portions within 2 voxels of the displayed principal slice plane or oblique
  slice plane. The sparse direction angular-error panel
  is intentionally not shown because it is too sparse to be useful for routine
  inspection. The predicted/fitted CP direction overlay is drawn as a thin
  anti-aliased line whose length is scaled by the in-slice projection magnitude,
  so out-of-slice directions are visibly shorter.
- The 3D target-presence panel in TensorBoard is display-only max-pooled with a
  `3x3x3` kernel before slicing. This must not modify `presence_target` used by
  training or test loss.
- 3D training/test loss logging reports average selected-branch direction
  angular error in degrees as `train/angle_mean_deg` and
  `test/angle_mean_deg`. The scalar is computed over sparse supervised
  direction samples with Lasagna 3x2 analytic decoding and unoriented
  `abs(dot)` agreement. Branch routing diagnostics include branch usage
  fractions and selected score means.
- 3D presence loss uses equal total class weight when routed positives and
  global negatives are both present:
  `0.5 * mean(selected positive BCE) + 0.5 * mean(all-branch negative BCE)`.
- When `training.test_interval > 0`, 3D training runs the configured test
  evaluation at step 0 before the first optimizer step and logs the same
  TensorBoard scalars/stdout as interval tests.
- Dense 3D test loaders do not inherit train augmentations by default.
  `training.test_augment_enabled: true` is the explicit opt-in for augmented
  dense tests.
- Dense 3D tests default to evaluating every configured held-out CP once in the
  deterministic pseudo-random test stream from sample index zero.
  `training.test_control_points: 0` is the explicit full-test sentinel with
  the same behavior. Positive values keep the fixed deterministic random test
  range beginning at `test_start_sample_index`.
- `python -m vesuvius.neural_tracing.fiber_trace_3d.train` is the 3D training
  entrypoint. It supports normal training, `--benchmark`, `--load-only`, and
  `--prefetch`.
- Normal 3D training also supports `--resume <snapshot.pt>`. The CLI path
  overrides config resume keys, restores model and optimizer state, writes a
  fresh timestamped run directory, and records the effective resume path in
  TensorBoard config text. If finite `training.max_steps` is not greater than
  the checkpoint step, training must fail clearly.
- 3D training and `--benchmark --load-only` runtime loading use
  `torch.utils.data.DataLoader` worker processes when
  `training.loader_workers > 0`. Each DataLoader item is one complete
  `FiberTrace3DBatch`, not an individual CP patch, and PyTorch default
  collation is bypassed so the custom dataclass is not nested or reshaped.
- Each 3D DataLoader worker lazily constructs its own `FiberTrace3DLoader` and
  VC3D sampler state in the worker process. Worker outputs are CPU
  `FiberTrace3DBatch` objects; the main training process transfers the whole
  batch to `training.device` immediately before forward/backward. The old
  thread-backed `_OrderedBatchLoadPipeline` is not a supported 3D loading path.
- In 3D configs, omitted or `null` `volume_cache_memory_mib` means a
  Python-side default of 512 MiB per VC3D sampler/loader/worker, not VC3D's
  internal 8 GiB default. Explicit positive values override this cap. The
  generated 2D Trace2CP geometry loader used by 3D evaluation receives the same
  default when the 3D raw config leaves the key unset or `null`.
- 3D DataLoader workers must not materialize full dense direction/presence
  target tensors. Worker batches carry image/valid tensors plus compact target
  descriptors: CP-only samples carry local CP/tangent metadata plus
  visualization-only transformed line segments, and NML dense-line samples carry
  transformed output-space line segments with precomputed patch bboxes for
  supervision. Dense `presence_target` and `presence_mask` are created by
  `fiber_trace_3d.targets.materialize_targets(...)` in the main training process
  on the training device. Direction supervision is represented sparsely as
  `direction_indices_bzyx`, `direction_target_sparse`, and
  `direction_weight_sparse`; normal training must gather predictions at those
  supervised line/CP voxels instead of creating full-patch dense six-channel
  direction targets.
- For JSON/non-NML 3D samples, `target_segment_*` metadata is visualization
  context only. The materializer must filter dense line rasterization by
  `_TARGET_MODE_DENSE_LINE`, so CP-only JSON segments do not create dense
  presence or direction supervision. Their direction target is the transformed
  CP tangent applied only to the CP neighborhood. TensorBoard visualization may
  draw those visualization-only segments in the target/context presence panel,
  but that display-only raster must not be fed back into loss materialization.
- The GPU target materializer must preserve the existing label semantics:
  NML sources supervise direction/presence by drawing the overlapping clipped
  fiber centerline voxels only, without a radius-expanded distance-to-segment
  tube. Non-NML sources supervise only the sampled CP neighborhood using
  `presence_radius_voxels`; that radius does not apply to NML centerline
  targets. Presence edge masking applies only to CP-only samples; NML dense-line
  samples supervise presence over the full valid patch. Lasagna 3x2 direction
  encoding uses the shared NumPy/torch-compatible helper semantics.
- `training.loader_workers` controls 3D DataLoader worker process count.
  `0` is the explicit serial/debug path. `training.loader_prefetch_factor`
  maps directly to PyTorch DataLoader prefetch factor for worker processes.
  `training.loader_worker_device` defaults to `"cpu"`. CPU worker processes
  use a guarded `forkserver` multiprocessing context where available, falling
  back to `fork` only when needed; CUDA worker devices select `spawn`.
- 3D `--benchmark --load-only` timing output separates main-process
  `wait_ms` from `to_device_ms`. It also reports worker-side profiling columns
  for loader construction, descriptor lookup, augmentation parameters,
  geometry-map creation, coordinate conversion, valid-mask generation, VC3D
  sampling, tensor conversion, value augmentation, compact target-spec
  generation, batch stacking, worker wall time, and worker CPU time. Dense
  target work is reported separately as main-process GPU target materialization
  timings (`target_ms`, `gpu_ms`, `line_idx`, `cp_idx`, `scatter`, `dir_enc`,
  `gpu_mask`, `linePts`, `dirPts`, and `posK`). With worker processes, the
  first `loader_workers` benchmark rows can include worker-local loader
  construction and should not be used as steady-state throughput.
- 3D prefetch computes chunk dependencies from the same explicit coordinate
  path used by training and the VC3D sampler. It follows the 2D step-count
  sentinel rules: omitted `--prefetch-steps` uses `training.max_steps`;
  positive values override config; explicit `--prefetch-steps 0` means every
  selected training CP once; negative values fail clearly. A positive
  `training.max_sample_index` bounds the prefetched training prefix, and
  full/config-driven prefetch also covers held-out test CPs once in flat order
  when `test_datasets` is configured.
- VC3D dependency collection currently accepts 2D coordinate surfaces shaped
  `[H,W,3]`. When 3D prefetch has a regular coordinate volume shaped
  `[Z,Y,X,3]` or another higher-rank `[...,H,W,3]` grid, the sampler wrapper
  flattens it into one 2D surface using the same convention as training
  sampling (`[Z,Y,X,3] -> [Z*Y,X,3]`), collects dependency metadata once, and
  de-duplicates chunks by `(store_identity, key)`. It must preserve VC3D
  returned metadata rather than reconstructing cache paths in Python.
- 3D prefetch must follow the same streaming dependency/download state machine
  as 2D prefetch: bounded dependency producers controlled by
  `prefetch_sampler_workers`, bounded download workers controlled by
  `prefetch_workers`, deterministic raw-sample-order producer consumption,
  global chunk de-duplication, cache-hit / `.empty` classification before
  downloads, earliest-raw-sample download priority, safe-prefix `idx`
  tracking, live dependency and download progress, sample skip accounting,
  fatal cancellation of queued futures, temporary PyTorch CPU intra-op thread
  pinning, and the shared Python atomic download helper.
- The only intentional 3D differences from 2D prefetch are that one 3D sample
  produces one CP-centered 3D augmentation-envelope dependency volume, valid
  counts are voxels, and there is no strip-z offset loop or top-view branch.
- V0 3D prefetch uses VC3D chunk dependency metadata and the shared Python
  prefetch writer with atomic cache-file renames and `.empty` marker handling.
  It does not prefetch Lasagna manifest channels.
- The 3D-to-2D evaluation bridge in `fiber_trace_3d.trace2cp_bridge` samples
  dense 3D model outputs at explicit 2D Trace2CP strip coordinates, projects
  six-channel Lasagna 3x2 direction predictions into the requested 2D strip
  frame, carries presence through, and reuses the existing 2D Trace2CP scorer.
  This bridge is metric/debug tooling only and does not change 3D input
  loading into strip loading.
- 3D Trace2CP projection must decode the six Lasagna 3x2 direction channels
  analytically: each two-channel projection is decoded with
  `theta = atan2(sin2theta, cos2theta) / 2`, then the three projection planes
  are reconstructed/sign-aligned with the Lasagna three-plane logic. Unit-sphere
  candidate tables, binned direction lookup, or grid-search decoding are not
  allowed for 3D Trace2CP projection.
- 3D training test evaluation may reuse the 2D `FiberStrip2DLoader` only to
  construct Trace2CP segment geometry. It must keep normal 3D training samples
  as CP-centered volume blocks. For Trace2CP evaluation, dense 3D inference is
  run over tiled axis-aligned blocks covering the requested 2D strip
  coordinates plus configured context, then sampled/projected back to 2D.
- When a 3D config defines `test_datasets` and `test_trace2cp_enabled` is
  false, test evaluation runs ordinary 3D sparse direction/presence loss on the
  held-out CP-centered 3D samples. It must not require Trace2CP geometry or
  trace loss.
- Configured dense 3D tests log `test_sample_3d/principal_slices` with the same
  principal-slice sheet layout as training at step 0 and interval test runs.
  The TensorBoard writer is flushed after configured test logging so initial
  test scalars and images are visible promptly.
- The `train_s1a_nml_all_64_sd2.json` 3D config includes the same held-out 2D
  fiber JSON `test_datasets` block as the full S1A NML 3D config, so step-0 and
  interval dense 3D test loss run for the fast 64-scale training setup.
- When `training.test_trace2cp_enabled` is true, 3D training logs
  `test/trace2cp_error`, raw y-error, valid segment count, and skipped segment
  count. `best.pt` and `current.pt` store `metric_name`; best-checkpoint
  selection uses `test/trace2cp_error` when it is available.
- `training.test_trace2cp_control_points: 0` means the full held-out Trace2CP
  CP set in flat order. Positive values use the deterministic random held-out
  range beginning at `training.test_trace2cp_start_sample_index` or, when that
  key is omitted, `training.test_start_sample_index`.
- The 3D Trace2CP metric path performs no training augmentations. Required 2D
  metric geometry must be explicit through `training.test_trace2cp_loader_config`
  or the 3D config keys `test_trace2cp_patch_shape_hw`,
  `test_trace2cp_strip_z_offset_count`, and `test_trace2cp_strip_z_offset_step`;
  missing required geometry must fail loudly.
- `python -m vesuvius.neural_tracing.fiber_trace_3d.train --trace2cp-vis`
  runs the same 3D projection/scoring path for one sample or a whole
  `--fiber-json`, prints `trace2cp_error=...` or `trace2cp_error_mean=...`, and
  exports `trace2cp_3d_vis.jpg`.
- `python -m vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool` is a
  separate native 3D Trace2CP inspection tool. It must not replace the
  projected `test/trace2cp_error` training metric or best-checkpoint selection
  unless that migration is explicitly requested.
- Native 3D Trace2CP selection supports both the existing
  `--sample-index`/`--target-offset` mode and explicit fiber segment mode:
  `--fiber-json <path> --start-cp-index A --target-cp-index B`. Explicit CP
  index mode requires `--fiber-json`, requires both CP indices, uses flat
  single-fiber CP ordering, and must reuse the existing 2D Trace2CP segment
  source builder with `target_control_point_index`.
- When `--fiber-json <path>` is supplied without explicit CP indices, native
  3D Trace2CP defaults to whole-fiber mode. Whole-fiber mode traces
  consecutive CP pairs from CP `0` to the last CP. Supplying both explicit CP
  indices keeps the single-segment debug mode; supplying only one CP index
  must fail loudly.
- The native 3D tool traces in selected-level ZYX voxel coordinates. It loads
  the same dataset/test-dataset CP pair as the visualization geometry loader,
  decodes six Lasagna 3x2 direction channels analytically, treats predicted
  axes as sign-ambiguous, and aligns sampled directions to the current trace
  direction before scoring.
- Native 3D Trace2CP inference uses overlapped axis-aligned model-output
  blocks. Each block has a full input patch and a cropped trusted core; point
  lookups must route to a block whose trusted core contains the queried point.
  The tool must not silently score candidates from cropped-away model-output
  borders.
- When the native 3D Trace2CP checkpoint output has grouped
  direction/presence branches (`7*K` channels), inferred block sampling decodes
  all `K` Lasagna 3x2 direction branches plus their branch-local presence
  values. Branch 0 remains the compatibility layout for single-branch callers,
  but native tracing must not be branch-0-only for grouped outputs.
- Native 3D Trace2CP cached inferred blocks must be CPU-resident. CUDA is used
  for model inference and transient block sampling, but the cache must not keep
  all inferred block output tensors on GPU across a long trace or strip render.
- Native 3D Trace2CP inference blocks are sampled through the configured
  `CoordinateSampler` using the same selected-level to base-coordinate
  conversion as 3D training: selected-level block grids are multiplied by
  `record.volume_spacing_base`, validated against `record.base_shape_zyx`, and
  passed to blocking `sample_coord_batch(...)`. Real configured volumes must
  not be read by direct zarr/raw block slicing in the native tool. These
  native block samples must use the strict requested-level VC3D blocking
  semantics above and reject reported fallback or chunk-error stats.
- Native 3D Trace2CP applies the configured 3D model-input normalization before
  inference. Exported native strip volume panels must display that same
  normalized input domain so the visualization shows what inference sees. For
  `image_normalization: "zscore"`, display maps a fixed normalized `[-3, 3]`
  window to `0..255`; for `minmax`, display maps normalized `0..1`; for
  raw/none modes, display clips raw `0..255`. Per-panel percentile display
  scaling is not allowed for native Trace2CP volume panels because it hides
  loading and brightness problems.
- Trace2CP strip rendering must reject non-blocking coordinate samplers and
  VC3D sampler results that do not report strict requested-level blocking
  semantics. Scale fallback, unresolved requested chunks, or chunk errors must
  fail loudly for debugging renders instead of being shown as valid strips.
- Native 3D Trace2CP defaults to `--inference-patch-shape-zyx 64 64 64`,
  matching the current fast 3D training/debug patch size, and
  `--core-margin-voxels 20` to crop away block-edge inference artifacts.
  Larger patch shapes remain explicit CLI overrides.
- Native 3D Trace2CP ordinary single-sample CLI mode defaults to sample index
  13 when no explicit `--sample-index` is provided. Bare `--fiber-json`
  without sample/CP selectors remains whole-fiber mode and must not be turned
  into sample-index mode by this default.
- Native 3D Trace2CP does not default to a fixed large step count. The default
  trace guard is distance-derived:
  `ceil(max_step_factor * cp_distance_voxels / step_voxels)`, with
  `--max-step-factor 3.0`. `--max-steps N` is only an optional additional
  safety cap.
- Native 3D candidate stepping samples deterministic tangent-plane angular
  offsets around the current inferred 3D direction. The default cone is
  `--cone-angle-degrees 25.0` with `--cone-angle-step-degrees 5.0`, keeping
  offsets inside the cone disk and always including the center direction. This
  produces 81 candidates at the default settings. The legacy square-grid
  generator is used only when `--cone-angle-step-degrees <= 0`, in which case
  `--cone-grid-size` controls the grid. Ring/azimuth candidate generation is
  not supported.
- Native 3D Trace2CP uses beam search by default. `--beam-width 8` keeps
  multiple cumulative candidate histories, `--beam-prune-distance-voxels 1.0`
  merges near-duplicate live beam states, and `--beam-lookahead-steps 1`
  expands short future trees before pruning. Pruning happens after the
  configured lookahead expansion, not after every single candidate step.
  `--beam-width 1` preserves the previous greedy one-step-commit control flow
  and bypasses lookahead. When target-plane candidates are found, the reached
  beam with the lowest cumulative score is selected; if no beam reaches the
  target plane before the step guard, the best live state is returned with the
  same failure reason semantics as greedy tracing.
- Native 3D beam-mode candidate selection is vectorized across the active
  beam/frontier states and their candidate directions for each lookahead
  depth. Candidate directions are generated as torch tensors on `cache.device`;
  current-point branch selection, candidate scoring, target-plane crossing, and
  pruning operate on tensors. Candidate points are then grouped by trusted
  inference block, sampled with batched `grid_sample`, decoded with the
  analytic Lasagna 3x2 torch decoder, and scored as tensors. The lazy
  inferred-block cache remains CPU-resident by design, so point-to-block
  routing and cache-miss block construction may still happen on CPU before
  the sampled tensors return to `cache.device`.
  For multi-branch outputs, every candidate evaluates every branch at the
  candidate point and uses the branch with the best score. Candidate selection
  minimizes a cost. By default, the direction score uses all-pairs product
  scoring over four signed/aligned directions: previous step direction,
  current-point sampled direction, candidate step direction, and candidate-point
  sampled direction. Candidate-sampled axes are sign-aligned to the candidate
  step direction, pairwise dots are clamped to `[0, 1]`, and the score is
  `presence * product(six pairwise dots)`. `--no-all-pairs-direction-product`
  restores the older two-dot score
  `dot(current_dir, step_dir) * dot(candidate_dir, step_dir) * presence`.
  `--candidate-substeps 1` is the default and preserves endpoint-only candidate
  scoring. With `--candidate-substeps S` for `S > 1`, candidate scoring samples
  the segment at `t = 1/S, 2/S, ..., 1`, evaluates all branches at every
  substep, takes the best branch score per substep, averages those substep
  scores, and then applies the current-point direction gate when legacy
  two-dot scoring is enabled. A multi-substep candidate is valid only when
  every substep has at least one valid branch. Search smoothness defaults to normal-aware split
  smoothness in the native 3D CLI. Candidate Lasagna normals are sampled
  directly at the candidate trace coordinates by converting selected-level ZYX
  points to base ZYX with `record.volume_spacing_base` and calling the existing
  batched Lasagna normal sampler/decoder used by the 2D geometry loader. The
  implementation must not interpolate normals by reference-line progress and
  must not reimplement Lasagna normal decoding in the candidate scorer. With a
  valid candidate normal, smoothness is split into tangent-plane turn and
  normal-tilt turn: tangent-plane turn is the angle between previous and
  candidate step directions after projection into the plane perpendicular to
  the Lasagna normal, while normal-tilt turn is the absolute elevation change
  against that normal. Both components use
  `max(0, angle - smoothness_free_angle)^2`, in radians, and the native 3D
  CLI default for `smoothness_free_angle` is `0` degrees so all measured
  turns are penalized unless explicitly overridden. The Lasagna normal
  sign ambiguity must not affect this penalty. The CLI flags
  `--smoothness-tangent-weight` and `--smoothness-normal-weight` override the
  component weights independently; their native 3D CLI defaults are `10.0` for
  tangent-plane turn and `0.1` for normal-tilt turn. If candidate normal
  sampling is unavailable or invalid for one candidate, that candidate falls
  back to the previous isotropic smoothness term
  `smoothness_weight * max(0, angle(previous_step_dir, step_dir) - free_angle)^2`.
  Native 3D Trace2CP also adds cumulative tangent-only smoothness over a
  short history direction so several small tangent-plane turns cannot compound
  into a large tangent-plane bend. This cumulative term is additive
  smoothness, not a direction/presence gate. It uses
  `--cumulative-smoothness-steps` to update a running trace heading and
  `--cumulative-smoothness-tangent-weight` to penalize the tangent-plane angle
  between that heading and the candidate step. It never penalizes
  normal/elevation change. If candidate normal sampling is unavailable,
  invalid, or tangent projection is degenerate, the cumulative term is zero for
  that candidate.
  The native 3D tool does not expose additive direction/presence
  candidate-selection weights.
- The first native 3D Trace2CP search step is seeded from the adjacent
  CP-local fiber-line tangent in the direction of the target CP's line index.
  It must not use the straight CP-to-CP chord and must not use the sampled
  model direction at the start CP. Forward and backward traces receive their
  respective initial directions from their start/target order. The first
  accepted step disables smoothness and evaluates the CP-tangent agreement only
  by the Lasagna-normal/elevation component at the candidate point; tangent
  plane rotation away from the CP tangent is ignored for that root expansion.
  In default all-pairs scoring, root-step pair terms involving the previous
  step/current CP tangent and candidate-sampled direction are neutralized so
  they do not reintroduce a tangent-plane CP-tangent penalty; the candidate
  sampled direction is still compared to the candidate step direction.
  The normal/elevation gate must be invariant to the Lasagna normal sign
  ambiguity. If the candidate normal is invalid or unavailable, that candidate
  falls back to the regular full `dot(current_dir, step_dir)` gate. Later
  steps use the sampled model direction at the current trace point,
  sign-aligned to the previous accepted step, and keep the full direction gate
  plus normal-aware smoothness. For multi-branch outputs, the current-point
  branch is chosen by best `dot(branch_dir, previous_step_dir) * branch_presence`.
- The native 3D CLI prints live progress bars for forward and backward tracing.
  Progress is measured by signed target-plane progress along the initial
  CP-to-CP direction. It includes step count, ETA, and inferred-block count.
- Native 3D strip visualization prints live progress for rendering stages and
  for side/top presence-strip sampling. Presence progress must report
  processed inference blocks, total unique inference blocks, sampled strip
  points, valid output points, newly inferred blocks, cached blocks, and total
  cache block count. Regular trace candidate sampling remains quiet unless a
  caller explicitly supplies a progress label.
- Native 3D strip visualization progressively overwrites the regular
  `trace2cp_native_3d_vis.jpg` output at render start, stage start/end, and as
  panels are rendered and added to the sheet. Before the first panel is
  available, the file must contain a status canvas rather than being absent.
  There must not be separate partial snapshot filenames; the same output path
  should always show the latest available status, partial sheet, or final sheet.
- `--trace-step-limit N` is a debug-only cap on accepted trace steps per
  direction. When set, native tracing can intentionally return a partial trace
  with `reason=trace_step_limit`; this is distinct from the safety guard
  `--max-steps`.
- Native 3D tracing stops by intersecting the plane through the target CP with
  normal from start CP to target CP. The returned trace appends the exact
  linear interpolation point on that target plane when crossing occurs.
- In native 3D whole-fiber mode, `--fiber-json <path>` without sample or CP
  selectors traces the entire fiber. `--fiber-json <path> --sample-index N`
  remains single-segment inspection using deterministic flat sample selection,
  and explicit `--start-cp-index/--target-cp-index` remains explicit
  single-segment inspection.
- In native 3D whole-fiber mode, each segment targets the plane through the
  next CP with the local CP-to-CP segment direction as plane normal. A segment
  succeeds only when the trace reaches that plane within the segment's step
  budget and the in-plane selected-voxel error to the target CP is at most
  `--whole-fiber-error-threshold-voxels` (default `100`). Successful segments
  continue from the reached crossing and carry the accepted trace direction.
  Failed segments count one restart and resume tracing from the failed target
  CP with a fresh CP-local fiber tangent.
- Native 3D forward/reverse fusion must preserve each trace's traced order.
  It must not sort points by CP-axis progress and average them. Fusion selects
  a forward/reverse point pair over traced arc length, not straight CP-axis
  overlap progress. Candidate score is
  `3D_pair_gap * 2.0 + forward_arc_length_to_pair + reverse_arc_length_to_pair`.
  Exact ties prefer smaller pair gap, then a more balanced/later meeting where
  both traces have traveled farther. The selected pair midpoint is the fusion
  meeting point: the forward start-to-meeting and reverse target-to-meeting
  partial traces are warped to that midpoint by traced arc-length fraction,
  concatenated, then arc-length-resampled as the CP-to-CP fused line.
  `closest_progress` is only a diagnostic projection of the selected midpoint
  onto the straight CP axis. Native Trace2CP reports failure only when no
  finite forward/reverse pair can be selected.
- Native 3D Trace2CP reports tool-local debug metrics:
  `native_trace2cp_plane_error` and
  `native_trace2cp_closest_target_error`, plus fusion diagnostics such as
  selected diagnostic progress, raw gap, considered pair score, and center
  penalty. For pairwise traced-arc fusion the center penalty is fixed to `1.0`.
  These are not the public 2D `trace2cp_error`.
- Native 3D whole-fiber mode reports its tool-local metric on a single line as
  `native_trace2cp_fiber_restart_rate=... restarts=... segments=...`, where
  the restart rate is `restart_count / segment_count`. Its JSON summary stores
  per-segment status, reason, reached-plane flag, in-plane error, step count,
  restart point, and reference arc distance at the last successful CP plane.
- Native 3D single-pair visualization first builds the initial side/top strip
  source from the existing 2D Trace2CP geometry loader for the input CP pair.
  In single-pair mode, the configured cross-strip height is a maximum cap: the
  rendered cross height is the odd centered size needed to cover the projected
  forward, backward, and fused traces with 50% extra margin, capped by that
  configured maximum. This adaptive render height is visualization-only and
  must not affect tracing or metric values.
- Trace2CP segment-source construction may trim extra line-window margin to the
  valid compact-geometry interval that contains both start and target control
  points. It must not synthesize missing normals. If the actual CP-to-CP line
  range crosses an invalid compact-geometry gap, source construction must fail
  loudly.
- Native 3D visualization includes side/top strip panels of the inferred 3D
  presence signal sampled on the displayed side/top strip coordinates from the
  native inference cache. Presence sampling should batch strip coordinates per
  strip rather than call model inference per pixel.
- Native 3D whole-fiber visualization uses four stitched panel rows: side
  volume, side 3D presence, top volume, and top 3D presence. Whole-fiber mode
  renders restart-delimited continuous long strips instead of one visual column
  per CP segment. Each visual span starts at the first CP after a restart and
  ends at the latest traced target CP in that span. Failed segment overlays are
  cut before they overlap the next CP region, then the displayed trace resumes
  from the restart CP in the next visual span. Whole-fiber visualization always
  uses a fixed 64 px cross-strip width; this width is visualization-only, and a
  traced path leaving the 64 px strip must only clip the drawn overlay, not
  invalidate tracing, metric calculation, or 3D sampling. The regular
  `trace2cp_native_3d_vis.jpg` path must be overwritten after every completed
  segment so long whole-fiber runs show partial visual progress at the final
  output filename.
- Native forward, reverse, and fused 3D traces are projected onto the initial
  side and top strip coordinate systems for overlay. The same visualization
  also rebuilds side/top strip geometry from the fused native 3D line and
  renders fused-line volume/presence panels with only the fused line overlaid
  thinly. The fused-line panels are debug visualization only; they do not define
  a new scoring path.

- The initial implementation loads batches of fiber-strip patches around random control points from the fiber dataset.
- Fiber source parsing accepts existing VC3D fiber JSON files and Knossos /
  WebKnossos `.nml` files. VC3D JSON parsing follows
  `vesuvius.neural_tracing.fiber_trace.fiber_json`.
- NML parsing orders nodes by edges, not XML order. Each usable open simple
  path component becomes one normalized `Vc3dFiber`; branch components, closed
  loops, disconnected singleton nodes, or malformed components are skipped or
  rejected with diagnostics rather than guessed through.
- NML line points and control points initially use the same ordered node
  coordinates unless a later explicit control-point convention is added.
- Each selected control point must be an exact member of `line_points`; otherwise the fiber JSON is rejected as inconsistent.
- The loader works on 2D sampled fiber side-strip patches.
- Neighboring strip-z context is represented as separate 2D patches.
- The default strip-z offset settings are `strip_z_offset_count=16` and `strip_z_offset_step=1.0`, generating `-7..8` selected-scale offsets and giving 16 patches per selected control point.
- Lasagna normals are used where needed to construct aligned strip frames.
- At loader startup, the loader builds one shared compact in-RAM fiber-line
  geometry store for all configured records. It computes the line-index ranges
  that can affect configured CP source windows and consecutive CP-to-CP
  Trace2CP spans, samples Lasagna normals only for those required ranges,
  builds valid contiguous frame intervals, and keeps compact per-line/frame
  arrays read-only for the rest of the process.
- A requested Trace2CP CP-to-CP span must not fail because an interior
  centerline point was omitted from compact-geometry preload. If compact
  geometry reports an invalid unsampled point inside such a span, that is a
  diagnostic bug path and the error must say so explicitly with a direct
  Lasagna value probe.
- Startup compact geometry construction may parallelize across independent
  records with process workers controlled by `loader_workers`.
  `loader_workers=0` means all logical CPU cores, and `loader_workers=1` is
  the serial startup/debug path. Each process opens its own base-volume and
  Lasagna channel handles, builds compact geometry for assigned records, and
  returns only compact arrays/metadata to the parent. Parallel startup may
  complete records out of order internally, but the final parent-owned store
  must be indexed by original record order.
- Startup Lasagna normal sampling may use batched/vectorized channel reads and
  normal decoding, but it must preserve Lasagna `_decode_normals`, ambiguous
  normal principal-axis handling, and strict invalid-data semantics.
- The compact geometry store is assembled and owned by the parent process, then
  shared by all threaded loader workers and cloned loaders in that process.
  Startup process workers must not remain as runtime geometry owners.
  `fiber_trace_2d` training does not currently use DDP or
  `torch.distributed`; this task must not introduce per-worker duplicated
  compact geometry.
- Runtime side/top source-grid construction looks up the record/control-point
  entry directly, evaluates only the requested source columns from the compact
  frame arrays, and broadcasts rows by frame axes. It must not write/read a
  dense per-CP coordinate cache.
- If a line point required by a CP source window cannot be sampled from the
  Lasagna manifest channels, the loader must not fabricate or propagate a
  replacement normal. That CP is invalid and is skipped by training/prefetch in
  deterministic stream order.
- During prefetch and training batch assembly, invalid CP-local samples caused by Lasagna channel data such as missing samples or in-bounds `grad_mag == 0` are skipped and reported, then the deterministic sample stream advances to the next sample.
- Fatal prefetch/training errors are infrastructure or programming failures such as missing APIs, broken bindings, interrupts, memory errors, or unexpected internal exceptions; those should stop the run rather than being hidden as data skips.
- VC3D side-strip/surface/segment sampling semantics define patch coordinates.
- Strip centerlines are sampled from all `line_points` with cubic Hermite interpolation over arc length; control points only select the strip anchor.
- The coordinate construction must be equivalent to VC3D side strips; flat planar patch simplifications are not acceptable except where they match the VC3D algorithm for that case.
- The implementation should reuse/export VC3D side-strip coordinate APIs when possible, or port the same algorithm with only small rounding/interpolation differences.
- Source-strip coordinate generation may use torch vectorization for Hermite
  interpolation and normal interpolation, but it must preserve the existing
  VC3D/Lasagna frame construction semantics.
- Dense source-strip coordinates, strip-z offset coordinates, geometric coordinate augmentation, and transformed line/control-point coordinates stay as torch tensors on the configured augmentation device until an explicit NumPy consumer boundary.
- The explicit NumPy boundaries are VC3D coordinate sampling, runner/PIL visualization/export, and sample metadata arrays. The loader must not repeatedly convert coordinates between NumPy and torch inside one source/augmentation path.
- The augment-vis source/patch path is the canonical loader path for runner exports, training batch loading, and prefetch coordinate generation.
- Augmentation visualization, training, runner batch loading, and prefetch must share the same CP-local source-strip and final-coordinate generation implementation.
- Normal training, benchmark/profile/load-only, augment-vis, line-trace-vis,
  dir-vis, and direct runner/debug center-patch loading use the configured
  `augment_device` for torch coordinate generation. With the example config,
  `augment_device: "auto"` uses CUDA when available. Prefetch dependency
  generation is the exception and stays CPU-pinned.
- The loader builds source geometry from the compact in-RAM store for each
  selected CP and reuses that source across augmentation variants and strip-z
  offsets as appropriate. Source-space line pixel coordinates remain the full
  per-column centerline used by training visualization, while volume sampling
  coordinates come from compact frame interpolation.
- Image loading samples base-volume Zarr values from explicit coordinates.
- Training/export coordinate sampling uses the VC3D blocking coordinate sampler: required chunks are collected and fetched/decoded before sampling, so a cold cache miss must not become an invalid output pixel.
- Interactive/progressive VC3D `tryGetChunk` semantics are not acceptable for this loader path because they can queue I/O and return an all-invalid first sample.
- `base_volume_scale` selects both the Zarr level to read and the sampling pixel scale: by default, one output patch pixel advances by one voxel at that selected level.
- Internally, fiber control-point coordinates remain in base-volume coordinates, so a selected level `s` uses a patch pixel spacing of `2**s` base voxels before coordinates are divided for reading level `/s`.
- The loader must not use the existing neural-tracing crop-loading path for image loading.
- Training's multiple strip-z offsets are derived from one CP-local source geometry by offsetting along the strip normal/frame direction, not by rebuilding a separate coordinate-generation path.
- Runtime geometric augmentation work should be batched across strip-z offsets
  and patches where tensor shapes are compatible. The implementation should
  avoid many tiny per-patch GPU calls when the same operation can be expressed
  as one batched tensor operation without changing deterministic sample order
  or augmentation semantics.
- Runtime image sampling should batch each CP sample's strip-z coordinate stack
  through `CoordinateSampler.sample_coord_batch`. If no native sampler batch API
  is available, flattening `[strip_z,H,W,3]` to one larger coordinate image is
  functionally valid because every output pixel samples from explicit 3D
  coordinates; only request traversal and cache/chunk locality may change.
- Dataset and loader settings are specified in Vesuvius-style JSON.
- Config keys include `datasets`, `batch_size`, `patch_shape_hw`,
  `strip_z_offset_count`, `strip_z_offset_step`, `seed`, `loader_workers`,
  `prefetch_workers`, `prefetch_sampler_workers`, `volume_cache_dir`, optional
  `volume_cache_memory_mib`, optional `volume_io_threads`, and optional volume
  cache settings. `strip_coord_cache_dir` has been removed and must be rejected
  if present.
- Augmentation config keys include `augment_enabled`, `augment_device`, `augment_seed`, `augment_shift_x`, `augment_shift_y`, `augment_rotation_degrees`, `augment_shear_x`, `augment_shear_y`, `augment_scale_min`, `augment_scale_max`, `augment_smooth_offset`, `augment_smooth_offset_stride`, `augment_brightness`, `augment_contrast_min`, `augment_contrast_max`, `augment_gamma_min`, `augment_gamma_max`, `augment_noise_std`, and `augment_blur_sigma`.
- Default augmentation extrema are `+-patch_width/4` px horizontal offset, `+-patch_height/4` px vertical offset, `+-180` degree rotation, `+-1` px/px shear, `sqrt(0.5)x..sqrt(2.0)x` scale, smooth curve offset up to `+-8` px with 16 px control stride, `+-0.25` valid-range brightness offset, `0.5x..2.0x` contrast around the valid patch center, `0.5..2.0` gamma, valid-range-relative noise std up to `0.125`, and Gaussian blur sigma up to `2.0`.
- Geometric strip augmentations operate on strip coordinates before image sampling.
- Training, augment-vis, line-trace, Trace2CP, labels, TTA, and all core
  loader paths must never do geometric augmentations as image-space operations.
  No helper, function, or API on those paths may geometrically warp, rotate,
  scale, shear, translate, resize, or flip an already sampled image/tensor.
  Such geometric changes must be represented as coordinate manipulation before
  sampling/slicing the patch.
- The only image-space geometric exception is the `--dir-vis` diagnostic probe:
  it may apply pixel-perfect identity, flips, and 90-degree rotations to an
  already sampled center patch to inspect checkpoint robustness. That exception
  must not be reused for training data, labels, augment-vis, line tracing,
  Trace2CP, or TTA.
- Image-space operations after sampling are allowed only for value-only changes
  such as brightness, contrast, gamma, noise, and blur.
- Geometric augmentation builds an oversized strip-coordinate source area, maps output patch pixels into that source, and samples the volume once at the final augmented coordinates to avoid edge and image reinterpolation artifacts.
- Geometric augmentation map handling is centralized in one paired fused map
  object. In this spec, "fused map" means actual precomputed coordinate map
  tensors, not a shared bundle of transform formulas.
- The paired transform must be constructed once for the specific source shape,
  output shape, augmentation parameters, and torch device. It must store:
  `backward_map_xy` for output pixel -> source pixel sampling, and
  `forward_map_xy` for source pixel -> output pixel line/control-point lookup.
- Both geometric map directions must be built explicitly as paired concrete
  map tensors during augmentation construction. Runtime consumers must never
  invert one map direction into the other after the fact, including by
  nearest-neighbor searches, brute-force distance scans, iterative solvers, or
  analytic/formula re-evaluation. If a runner, loader, target, or visualization
  path needs the opposite direction, it must receive and sample the prebuilt
  opposite map.
- Every geometric augmentation stage is baked into those map tensors at
  construction time: translation, flips, scale, shear, rotation, and smooth
  offset. No geometric augmentation may invert coordinates with rasterized
  masks, image warps, nearest-neighbor searches over the output grid,
  brute-force distance scans, iterative solvers, or runtime analytic inverse
  formulas.
- Smooth offset augmentation is a direct paired vertical map in source-strip
  coordinates. The output-to-source map applies the smooth offset as
  `source_y += f(source_x)`, and the source-to-output point map applies the
  inverse as `source_y -= f(source_x)` before the affine forward map. It must
  not require iterative solving or dense nearest-grid inversion. Smooth control
  generation/interpolation is allowed only while constructing the fused map
  tensors; it must not run during line/control-point lookup.
- Affine geometric shift is an output-space translation applied after scale/flip, not a source-space translation before scale. Combined shift+scale must keep image sampling, transformed line coordinates, and transformed control-point coordinates under that same composition.
- Training line targets and debug line overlays are geometric coordinate products, not raster images. The line must be represented by strip/output pixel coordinates after the same geometric coordinate transform used for image sampling.
- Transformed line/control-point coordinates are computed from cached
  source-space line/control-point coordinates by bilinear lookup/interpolation
  against the precomputed `forward_map_xy`. Smooth-offset line/control-point
  mapping must not run smooth interpolation, evaluate affine transform formulas,
  or invert the patch by dense output-grid nearest-neighbor search.
- Line points and the control point for a patch must be transformed together in
  one vectorized lookup call through the fused map object, then split back into
  line and CP outputs.
- Sparse line/control-point mapping must use direct bilinear gather against
  `forward_map_xy`. Tiny `grid_sample` calls for sparse point lists are not the
  intended implementation because their fixed launch overhead dominates the
  small amount of point data.
- When multiple patches from one loader sample need transformed line/control
  point coordinates, their `forward_map_xy` tensors and source point lists
  should be stacked and processed as a batched sparse lookup where shapes are
  compatible.
- Coordinate augmentation should stack `backward_map_xy` tensors and run
  batched dense sampling for compatible strip-z offset patches.
- When multiple strip-z offsets share the same CP source geometry and the same
  augmentation parameters, transformed line/control-point coordinates must be
  computed once and reused across those offsets.
- The line must never be transformed by resampling a raster line mask. No geometric augmentation may be implemented as an image-space transform of a previously rasterized line, mask, or image patch.
- Debug visualization may rasterize the transformed line coordinates only as the final drawing step, with fixed screen-space thickness/opacity, so line thickness and sharpness are not affected by scale, rotation, shear, or interpolation artifacts.
- Any future training target derived from the fiber line must use the same transformed output pixel coordinates as the sampled image, so labels and image pixels remain aligned exactly.
- Image/value augmentations after Zarr loading run as torch tensor operations on the configured device.
- Value augmentations after VC3D image loading should run as batched tensor
  operations where possible. Variable per-patch Gaussian blur uses grouped
  batched convolutions instead of one CUDA convolution loop per patch.
  Per-patch operations remain acceptable only when required to preserve behavior,
  such as deterministic per-patch noise streams.
- VC3D coordinate sampling remains the explicit image I/O boundary unless the
  sampler exposes a true batched coordinate API. The loader must not add a
  separate image sampling path just to batch around that boundary.
- Augment visualization uses raw clipped image values and must not apply percentile or per-cell normalization.
- The augment visualization mode renders a three-row JPG contact sheet: lower-limit examples, upper-limit examples, and random combined training-style examples.
- Augment visualization prints no timing diagnostics by default.
- `--augment-vis --augment-profile` enables timing diagnostics. Profile mode runs the same sample and augmentation entries twice, prints a pass 1 table for cold/first-use costs and a pass 2 table for warmed costs, and each table includes per-entry rows plus full total/average-per-patch summaries and `total/no-first` plus `avg/no-first` summaries that exclude the first unaugmented row.
- Augment contact sheets draw the transformed fiber-line coordinates at 50 percent opacity with fixed drawing thickness.
- Augment contact sheets draw a final visualization-only thin vertical marker at the transformed control-point coordinate for each patch, leaving a small gap around the CP pixel itself.
- Augment contact-sheet cells include a top label band naming the shown augmentation; labels must not overlay image pixels.
- Dataset entries include `fiber_paths` or `fiber_glob`, `base_volume_path`, `base_volume_scale`, and required `lasagna_manifest_path`.
- Dataset entries may define a fiber-coordinate affine transform using one of
  `fiber_transform_json` / `fiber_transform_json_path` for Vesuvius
  registration `transform.json`, inline `fiber_transform`, or Lasagna-compatible
  inline `transform`. Inline matrices are XYZ 3x4 or homogeneous 4x4. The
  matrix direction is source/moving fiber XYZ to current/fixed base-volume XYZ;
  `fiber_transform_invert` or `transform_invert` inverts it before use.
- Fiber-coordinate transforms are applied once immediately after JSON/NML
  parsing and before bounds checks, sample ordering, strip-coordinate cache
  identity, prefetch, training, and Trace2CP tooling. Lasagna manifest normals
  are still sampled from the current manifest after transformation.
- Optional top-level `test_datasets` uses the same dataset-entry schema as `datasets`; when present it defines a separate deterministic test loader while reusing the rest of the loader configuration.
- Strip-frame normals are sampled only through the Lasagna manifest `grad_mag`, `nx`, and `ny` channels.
- Trace2CP segment samples carry the line-window original line indices and the
  signed Lasagna normals used to build the segment strip, plus the actual
  start/target strip row-axis vectors after VC3D-style frame construction.
  Single-pair Trace2CP may choose either valid sign, but whole-fiber Trace2CP
  must align each later pair-local row axis to any already accepted shared-CP
  row-axis reference before sampling image data. If the initially built grid's
  row-axis disagrees with that reference, the loader flips the local Lasagna
  normal sequence and rebuilds the grid. This prevents adjacent segment images
  from flipping in y solely because Lasagna normals are sign-ambiguous.
- Normal batch loading samples control points in deterministic pseudo-random order from the configured seed.
- The deterministic pseudo-random training order covers every configured control point exactly once per dataset pass before repeating.
- Changing training step counts, batch size, or control points per step must only truncate or extend the consumed prefix of that deterministic sample stream; it must not reshuffle earlier samples.
- `load_batch` may parallelize CP-sample construction with `loader_workers`,
  defaulting to the machine logical CPU count. `loader_workers=0` explicitly
  requests all logical CPU cores. Parallel workers may evaluate candidate
  samples out of order, but accepted output samples and skip handling must
  follow the same deterministic sample-index order as serial loading.
  `loader_workers=1` is the serial no-thread debug path. When
  `loader_workers > 1`, the loader reuses a persistent CP-level executor across
  batches instead of constructing a new thread pool per step.
- The same `loader_workers` setting also controls startup compact-geometry
  record construction. Startup uses process workers, while warm
  `load_batch` CP-sample construction uses the persistent in-process thread
  executor above. No separate startup worker-count key exists.
- Parallel loader workers must not serialize on deterministic random-order
  locks during the warm path. Random dataset-pass orders are built once per
  pass, cached by pass index, and prewarmed for the attempted batch window
  before CP workers are submitted.
- The tester/runner loads a batch from a specified deterministic control-point sample index.
- Prefetch uses the same shared source-strip implementation as training and augment-vis.
- Prefetch remains CPU-pinned, but it still uses the same torch-native source-grid and strip-offset path, converting to NumPy only once for VC3D dependency discovery.
- Prefetch is independent of any one random augmentation draw: for each selected CP and strip-z offset it covers the configured maximum augmentation envelope represented by the oversized source-strip coordinates.
- Prefetch may conservatively cover more chunks than one concrete augmented training sample, but it should avoid misses for later random augmentations within the configured extrema.
- Prefetch must use dependency-only chunk discovery for the base-volume sampler. For VC3D this means `collect_coords_dependencies` over the same conservative source-envelope coordinates, without `sample_coords`, image-value sampling, or discarded sampled pixels.
- VC3D dependency discovery must return explicit per-chunk metadata for Python prefetch: remote chunk key/URL, final persistent cache data path, `.empty` marker path, persistent extension, and cache payload format.
- Python prefetch must not reconstruct VC3D persistent cache paths or remote chunk keys; it consumes the metadata returned by VC3D dependency discovery.
- Python prefetch currently supports only direct-source uncompressed chunks whose remote payload is exactly the persistent `.bin` payload VC3D expects. Compressed, filtered, sharded, byte-swapped, or otherwise non-direct payloads must fail clearly until explicit codec support is added.
- Normal image loading still uses the VC3D blocking coordinate sampler, so training/export samples are decoded before image sampling returns.
- Prefetch classifies VC3D persistent-cache files in Python: existing data files are cache hits, existing `<cache>/level_<level>/<iz>/<iy>/<ix>.empty` files are known-missing hits, and definitive missing chunks write that same zero-byte `.empty` marker.
- Prefetch data downloads are written to unique temporary files in the final cache directory and then atomically renamed to the VC3D-provided final cache path.
- VC3D also writes persistent-cache `.empty` markers as zero-byte files and reads them by existence.
- Prefetch performs global chunk deduplication by store identity and chunk key before network work.
- Prefetch runs parallel dependency producers plus bounded chunk download workers; download worker count is controlled by `prefetch_workers` without an additional hard-coded cap, and dependency/sampler producer count is controlled by `prefetch_sampler_workers`.
- During prefetch, PyTorch CPU intra-op threads are temporarily forced to `1`
  while dependency producers run, then restored. This prevents each producer
  from fanning out over the full machine and makes `prefetch_sampler_workers`
  the practical CPU-side source/dependency generation limit.
- Prefetch may generate dependency requests with parallel producers, but producer
  results are consumed in raw deterministic sample-index order before chunks are
  classified or enqueued for download.
- Prefetch schedules not-yet-submitted chunk downloads by the earliest raw
  deterministic sample index that requested the chunk. This keeps downloads as
  close as practical to `idx` order and avoids burying earlier-sample chunks
  behind a large later-sample executor backlog. Transfers already active are
  not cancelled or restarted for reprioritization.
- Prefetch reports sample/dependency progress only while dependency generation is incomplete; once all requested samples have been processed or skipped, live progress reports only download progress. The live progress includes unique chunks, cache hits, known-missing chunks, downloaded chunks, queued download futures, configured transfer worker count, configured sampler/dependency producer count, skipped samples, errors, and MiB/s. The download denominator is the number of chunks that were not cache hits or pre-existing `.empty` markers and therefore needed fetch/missing resolution. While dependency generation is incomplete, download ETA extrapolates from observed chunks per sample and observed cache-hit/known-missing/download-needed ratios.
- Prefetch reports skipped invalid samples separately from download errors and includes the first skip reason.
- If prefetch hits a fatal producer error, queued producer/download futures are cancelled so shutdown does not wait on a large stale download backlog.
- Prefetch remains base-volume-only; Lasagna manifest channels are not prefetched by the VC3D base-volume prefetch path.
- V0 training is provided by `python -m vesuvius.neural_tracing.fiber_trace_2d.train`.
- `train.py --prefetch` runs training-oriented chunk prefetch only and exits before model, optimizer, TensorBoard, run-directory, or snapshot setup.
- `train.py --benchmark` runs 100 training batches and exits without test evaluation, TensorBoard, run-directory creation, or snapshot setup. It reports throughput in CNN image patches per second, where one patch is one flattened strip-z image sent through the 2D model.
- `train.py --profile` runs the same 100-batch benchmark path with per-batch timing rows and a final average milliseconds-per-patch summary. Profiled stages are aggregate coordinate generation, descriptor lookup, strip-coordinate cache load, source geometry generation, line-coordinate generation, coordinate augmentation, base-volume Zarr read/sampling, torch image/value augmentation, forward plus loss, and backward plus optimizer step.
- When loader CP parallelism is enabled, profile rows report both loader wall
  time and summed loader worker time. The threading factor is
  `worker_time / wall_time`; stage columns such as descriptor/cache/load remain
  summed worker timings and must not be interpreted as wall time under
  parallel loading.
- Profile rows also report whole-process CPU time for the benchmark batch. The
  process CPU factor is `process_cpu_time / batch_wall_time`; compare this
  value against system CPU-utilization monitors when checking whether loader
  work is actually keeping CPU cores busy.
- `train.py --load-only` runs the same 100-batch benchmark loader path and exits without test evaluation, TensorBoard, run-directory creation, snapshots, image/value augmentation, image normalization, supervision building, model forward, backward, or optimizer work. It still performs deterministic sample selection, CP-local source construction, coordinate augmentation, and base-volume sampling so loading bottlenecks can be isolated. When `training.pipeline_enabled` is true, load-only benchmarks use the bounded whole-batch queue so loader parallelism can be measured without model work.
- Training and training prefetch use the same deterministic pseudo-random CP sample-index sequence: each pass visits all configured CPs once in seeded random order and wraps at dataset end.
- With `training.max_steps = 0`, training repeats the full training dataset indefinitely.
- `training.max_sample_index` is an optional positive exclusive deterministic sample-index limit. The default `0` means no limit. When positive, training wraps every global sample position with `sample_index % training.max_sample_index`, so long runs reuse that deterministic CP/data prefix independently of `training.max_steps`. The limit does not bound augmentation seeding: geometric and value/image augmentation draws are keyed by the unbounded training stream index so repeated use of the same bounded CP sample gets fresh deterministic augmentation parameters instead of replaying the same transform.
- Explicit positive `--prefetch-steps N` overrides `training.max_steps` and prefetches exactly `N * training.control_points_per_step` CP samples from the deterministic random training stream.
- Explicit `--prefetch-steps 0` overrides `training.max_steps` and prefetches every configured training-dataset CP once, independent of `control_points_per_step`; if `training.max_sample_index` is positive it prefetches that bounded deterministic prefix instead. When `test_datasets` is configured, it also prefetches every held-out test CP once.
- If `--prefetch-steps` is omitted, prefetch uses `training.max_steps`; if that configured value is `0`, omitted prefetch also means every configured training/test CP once.
- Negative `--prefetch-steps` values are invalid.
- The V0 trainer uses `FiberStrip2DLoader` batches directly; it must not use the neural-tracing 3D crop loader or a separate image sampling path.
- In training and benchmark modes, top-level `batch_size` is the number of CP
  samples loaded per step and must match `training.control_points_per_step`.
  Non-default flattened CNN patch counts are valid and must not warn; the
  flattened CNN patch count is `batch_size * strip_z_offset_count`.
- Training geometric augmentations are the same coordinate-space augmentations used by augment-vis. Value augmentations run through the existing torch augmentation functions after Zarr sampling.
- On CUDA training runs, `training.pipeline_enabled` may overlap future batch loading with current-batch model work through a bounded deterministic producer/consumer queue. The same queue is used by `--load-only` benchmarks to measure batch-loader parallelism directly. `training.pipeline_depth` controls the number of submitted whole-batch futures and defaults to `16`. `training.pipeline_workers` controls how many whole-batch `load_batch` calls may run concurrently and defaults to `8`; `0` means use `pipeline_depth`. Whole batches are still consumed strictly by step number, so the deterministic CP sample stream is unchanged.
- `training.pipeline_isolated_loaders` defaults to `false`. The normal
  pipeline shares the base loader, parsed fiber/Lasagna metadata, deterministic
  sample-order cache, and VC3D sampler/cache across whole-batch futures.
  Setting it to `true` creates worker-local loader clones with fresh VC3D
  samplers but shared parsed records and deterministic order cache.
- `volume_cache_memory_mib` is an optional VC3D sampler cache budget. `null` or
  omission leaves VC3D's default cache behavior intact. Positive values cap
  each VC3D sampler cache, which is mainly useful when
  `training.pipeline_isolated_loaders=true` duplicates VC3D samplers.
- `volume_io_threads` optionally forwards a positive VC3D I/O thread count to
  each VC3D sampler when the installed binding exposes that control.
- The training pipeline keeps strip-coordinate generation and geometric coordinate augmentation on the configured `augment_device`. It does not move those torch coordinate operations to CPU. The only unavoidable CPU/NumPy boundary remains the VC3D coordinate sampler call after final coordinates are generated.
- The pipeline uses the shared `FiberStrip2DLoader.load_batch` path with image/value augmentation deferred. Loaded batches carry the deterministic per-patch augmentation parameters.
- CUDA training prepares loaded batches on a separate CUDA stream when `training.pipeline_enabled` is true. A background preparation executor submits deferred torch image/value augmentation, image normalization, and direction-supervision tensor construction for loaded batches, then records a CUDA event. The main training stream waits for that event immediately before forward pass.
- Normal CUDA training must keep prepared augmented image tensors on CUDA and must not round-trip deferred value augmentation through NumPy. Runner/debug APIs may still return NumPy batches for export and tests.
- Training timing logs include CPU batch load time, load-pipeline wait, preparation enqueue time, measured CUDA preparation time, preparation wait time, and preparation submit time for queuing future prepared batches. `prep_submit_ms` is main-thread queue-refill overhead; the full preparation work is represented by `prep_ms`/`prep_gpu_ms` and runs in the background preparation executor on CUDA pipeline runs. Profile mode also reports an `outside` aggregate for work outside the forward/backward/optimizer critical path.
- Normal training prints the effective CUDA pipeline enable flag, queue depth,
  whole-batch loader workers, and loader CP-worker count once at startup.
- Whole-batch loading may use multiple concurrent producers. Zarr cache tracing is thread-local, and whole-batch outputs remain ordered at consumption time.
- CPU training runs default to the synchronous path even when `training.pipeline_enabled` is true; the automatic pipeline is a CUDA training optimization.
- A training step samples `training.control_points_per_step` deterministic control-point samples and every configured strip-z offset. The default is four control points and 16 strip-z offsets, giving 64 2D strip patches.
- If a deterministic training sample is invalid because its CP-local Lasagna normal window is invalid, batch loading skips it and continues with following deterministic sample indices until the requested number of control-point samples is loaded. If too many consecutive samples are invalid, batch loading fails with a clear error.
- Training flattens control-point and strip-z dimensions into a patch batch before the 2D model forward pass.
- The default V0 direction model is a 10-block residual CNN with 64 hidden channels. It uses a 3x3 input projection, constant-width residual blocks, BatchNorm2d normalization, a final 1x1 direction projection, an optional 1x1 sheet/fiber-presence projection, and an optional 1x1 embedding projection for explicit contrastive experiments.
- Standard training uses direction supervision plus sheet/fiber presence when
  `training.presence_enabled` is true. Contrastive embedding training is
  disabled by default and must be explicitly enabled.
- V0 model output always starts with exactly two per-pixel direction channels in the Lasagna ambiguous two-cos-channel encoding. When `training.presence_enabled` is true, one sigmoid sheet/fiber-presence channel follows the direction channels. When `training.contrastive_enabled` is true and `training.contrastive_embedding_channels > 0`, raw embedding channels are appended after direction and any presence channel. A disabled contrastive config must instantiate no embedding head, even if a stale positive `contrastive_embedding_channels` value is present. Consumers must use explicit output-slicing helpers instead of hard-coded embedding offsets.
- When `training.top_view_enabled` is true, training jointly instantiates a
  second V0 model for top-view strip slices. This top-view model outputs the
  same two Lasagna ambiguous direction channels plus one sigmoid scalar channel
  interpreted as a fiber-center distance transform, not sheet/fiber presence.
  The side model output layout is unchanged.
- For strip-image tangent angle `theta`, target channels are `0.5 + 0.5*cos(2*theta)` and `0.5 + 0.5*cos(2*theta + pi/4)`.
- Sheet/fiber presence training is enabled by `training.presence_enabled`.
  It supervises each loaded strip patch's rounded transformed CP pixel as
  presence `1`. Valid non-CP pixels inside the same shift-reachable
  CP-neighborhood rectangle used for contrastive negatives are supervised as
  presence `0`; unreachable patch edges are ignored so the network does not
  learn that CPs can never occur there. The positive-pixel BCE mean and the
  negative-pixel BCE mean have equal aggregate weight, and the combined loss is
  multiplied by `training.presence_weight`.
- Contrastive embedding training is experimental opt-in, enabled by
  `training.contrastive_enabled`.
  It requires `training.contrastive_embedding_channels > 0` and
  `training.control_points_per_step` divisible by
  `training.contrastive_control_points_per_fiber`.
- In contrastive mode, each training step loads deterministic same-fiber CP
  groups: every group contains `contrastive_control_points_per_fiber` CPs from
  one fiber, and consecutive groups are concatenated to fill
  `control_points_per_step`. Group ordering is deterministic and covers the
  effective CP set by shuffled fiber-local CP groups before repeating.
- Same-fiber CP patches keep independent geometric augmentation draws through
  unique raw sample indices. Value/image augmentation draws are synchronized
  within each same-fiber group so the embedding objective does not treat
  value-only appearance jitter as identity evidence.
- The contrastive embedding loss uses cosine similarity on each loaded strip
  patch's rounded transformed CP pixel. Positive terms are z-search-aware:
  for every anchor CP sample/strip-z offset, candidates are only other CP
  samples from the same fiber, across their loaded strip-z offsets. The
  already most-similar candidate is selected and trained toward cosine
  similarity `1`; same-CP offsets are not used as positives. Negative terms
  compare each CP embedding sample with one deterministic valid non-CP pixel
  from the batch and penalize cosine similarity above
  `training.contrastive_negative_margin`. Negative candidates are restricted to
  the CP-neighborhood reachable rectangle implied by the configured
  output-space `augment_shift_x/y` bounds; unreachable patch edges are ignored,
  not supervised as negatives. CP embeddings from other fibers are not used as
  negative samples. Positive and negative means are averaged equally, then
  multiplied by `training.contrastive_weight`.
- Contrastive embedding training also includes a similarity-image sparsity
  term. For each supervised CP embedding, the embedding similarity image
  against that CP is computed in normalized visualization space
  `0.5 + 0.5 * cosine_similarity`; its valid-pixel mean over the same
  shift-reachable CP area used for contrastive pixel negatives is trained
  toward the fixed target `0.1` with MSE. This term is added to the balanced
  positive/negative pair loss before applying `training.contrastive_weight`, so
  CP-similar embeddings are encouraged to stay spatially sparse without using
  unreachable patch edges as evidence.
- Presence visualization writes TensorBoard presence-probability maps when the
  presence head is enabled. Contrastive embedding visualization writes
  TensorBoard similarity maps:
  per-pixel cosine similarity against the selected patch's CP embedding is
  mapped from `[-1, 1]` to `[0, 255]` with invalid pixels black.
- Equivalent implementation formulas are `cos2theta=(dx^2-dy^2)/(dx^2+dy^2+eps)`, `sin2theta=2*dx*dy/(dx^2+dy^2+eps)`, `dir0=0.5+0.5*cos2theta`, and `dir1=0.5+0.5*(cos2theta-sin2theta)/sqrt(2)`.
- Lasagna two-channel direction decoding must use the analytic inverse:
  `cos2theta=2*d0-1`,
  `sin2theta=cos2theta-sqrt(2)*(2*d1-1)`, and
  `theta=atan2(sin2theta, cos2theta)/2`. Binned or candidate-angle lookup
  decoders must not exist anywhere in `fiber_trace_2d`.
- Forward/backward ambiguity comes from the double-angle encoding itself; `(dx,dy)` and `(-dx,-dy)` must encode identically.
- Direction targets are derived from the transformed output-pixel line coordinates produced by the same augmentation path as the image. They must not be derived from unaugmented line points for augmented patches.
- Each loaded strip sample carries the transformed control-point output-pixel coordinate. V0 direction supervision is limited to the eight neighboring pixels around that rounded transformed control-point location, filtered by image validity and patch bounds.
- The V0 loss compares predicted and target encoded channels directly with MSE over those CP-local samples; raw signed `(dx,dy)` regression and `abs(dot)` losses are not the V0 training representation. Training additionally reports folded unoriented angular error in degrees over the same supervised pixels, with `0` degrees perfect and `90` degrees maximally wrong.
- Top-view training loads one top-strip patch per loaded CP sample, using the
  same deterministic CP ordering and the same geometric/value augmentation
  parameters as that CP's center side-strip sample. The top strip is sampled
  with the VC3D-style `lineSurface`/top-strip coordinate construction already
  used by Trace2CP visualization: columns follow the fiber line and rows follow
  the side/cross-fiber axis derived from Lasagna normals.
- Top-view training uses the same cached, vectorized, batched loader mechanics
  as side-view training. The only source-coordinate difference is the grid
  builder: top view uses the top-strip builder, side view uses the side-strip
  builder. Top-view coordinate augmentation stacks maps and tensors through the
  same batched coordinate-resampling helper, and top-view image loading is
  grouped through `CoordinateSampler.sample_coord_batch`.
- Because a top patch and its center side-strip patch use the same source/output
  pixel frame and the same geometric augmentation parameters, the transformed
  fiber line and CP pixel coordinates are identical. Top-view batch loading
  must reuse the already computed side-sample line/CP coordinates instead of
  running a second line-coordinate lookup for the top patch.
- Top-view direction supervision uses the transformed top-strip line tangent
  and the same Lasagna ambiguous two-channel MSE objective as side strips.
  Top-view distance-transform supervision uses only the rounded normal
  cross-section through the transformed CP. Its target is `1.0` at the CP,
  falls linearly to `0.0` at `training.top_view_dt_radius_px` pixels
  (default `30.0`), and remains explicitly supervised as `0.0` for valid
  rounded-line pixels beyond that radius. The top direction and DT losses are
  multiplied by `training.top_view_direction_weight` and
  `training.top_view_dt_weight`.
- Training creates a run directory from `training.run_path` and `training.run_name` plus a date string. Passing `--resume <snapshot.pt>` creates and names a fresh run directory the same way, restores model and optimizer state from the snapshot, starts from `checkpoint_step + 1`, and keeps `training.max_steps` as the absolute target step. To continue past a finished run, increase `training.max_steps` before resuming. If two runs start in the same second, a numeric suffix is added to avoid a run-directory collision.
- Training config keys include `max_sample_index` for bounded deterministic-prefix reuse, `pipeline_enabled`, `pipeline_depth`, `pipeline_workers`, and `pipeline_isolated_loaders` for CUDA training load/model overlap, and `test_interval`, `test_control_points`, `test_start_sample_index`, `test_trace2cp_step_px`, and `test_trace2cp_rf_margin_px` for deterministic test evaluation when `test_datasets` is configured.
- Test evaluation runs at step 1, every `training.test_interval`, and the final step when `test_datasets` is configured. Positive `training.test_control_points` values load the fixed deterministic random range starting at `training.test_start_sample_index`, so the same held-out CP samples are compared across time. `training.test_control_points: 0` is the full-test sentinel: it evaluates every configured held-out CP sample once in flat CP order starting at zero, ignoring `training.test_start_sample_index`, so whole-fiber test metrics can be compared directly against `--trace2cp-vis --fiber-json` on the same held-out fiber apart from pair-alignment details. In addition to fixed-batch direction loss, the test path evaluates the public Trace2CP metric by tracing each selected held-out CP to its next CP segment and averaging valid `trace2cp_error` values.
- TensorBoard logging writes the training config JSON as text, direction-loss scalars, angular-error degree scalars, timing/cache diagnostics, and batch direction overlay images at configured intervals. Batch direction overlays show the transformed fiber centerline as context and one short network-predicted direction segment at the transformed CP; they do not draw CP-neighborhood supervision boxes or extra CP markers. Overlay contact sheets select examples across loaded control-point samples first, preferring each CP's strip-z offset closest to zero before showing additional offsets. When `test_datasets` is configured, TensorBoard also logs `test/loss_direction`, `test/angle_error_mean_deg`, `test/supervision_samples`, test cache diagnostics, and a `test/batch_direction_overlay` image at test evaluation steps.
- When top-view training is enabled, TensorBoard also logs top-view
  direction/angle/DT scalars and writes top-view image summaries: a GT-line
  plus predicted-direction overlay and a fixed `0..1` DT scalar map for train
  and test batches.
- Console training progress prints every one of the first 100 training steps,
  then falls back to `training.scalar_log_interval`.
- Prefetch progress includes `idx=<exclusive-index>` showing the largest
  contiguous exclusive deterministic training-stream prefix whose required
  chunks are cache-complete. This index is counted through the seeded shuffled
  CP stream used by training, before mapping a stream position to its original
  flat fiber/CP id. Operators can use that value as
  `training.max_sample_index` to train on the same prefetched random-prefix
  stream. A stream sample is cache-complete only after every dependency request
  for that sample has been classified and each required chunk is a cache hit, a
  known/new missing marker, or a completed successful download. Dependency
  generation alone must not advance `idx` while downloads are still pending.
  When `training.top_view_enabled` is true, prefetch dependency generation
  includes the top-view strip envelope in addition to all side-strip z-offset
  envelopes, and both views must be complete before that sample can advance the
  prefix.
- Training writes snapshots under `<run_dir>/snapshots/current.pt` and `<run_dir>/snapshots/best.pt`. With `test_datasets`, current snapshots are written at the test evaluation cadence and best is selected by lowest observed averaged `test/trace2cp_error`. Without `test_datasets`, current snapshots use `training.checkpoint_interval` and best is selected by lowest observed training loss. `training.kept_snapshot_interval` defaults to `10000` and writes retained numbered snapshots named `step_<iteration>.pt`; `0` disables retained numbered snapshots. A resumed run writes its own fresh `current.pt`, `best.pt`, and retained numbered snapshots under the newly created resumed run directory.
- The runner is `python -m vesuvius.neural_tracing.fiber_trace_2d.runner`.
- Augment contact sheets are exported with `--augment-vis --export-dir <dir>`. Add `--augment-profile` to print cold and warm augment timing tables.
- Direction-field inspection is exported with `--dir-vis --checkpoint <snapshot> --export-dir <dir>`.
- Direction-field inspection uses the same deterministic `--sample-index` ordering as training, prefetch, augment-vis, and line-trace-vis. It loads the center side-strip patch, center-crops it to the largest native square when needed, applies pixel-perfect image-space identity, flip-x, flip-y, rot90, rot180, and rot270 variants, runs the checkpointed direction model on each native-resolution variant, decodes the Lasagna ambiguous two-cos-channel output, nearest-neighbor scales each augmented patch image by 4x for visualization only, and draws short direction line segments on top.
- Direction-field inspection draws only every second source pixel in x and y, so each drawn sample corresponds to an 8x8 display-pixel cell in the 4x visualization. It draws anti-aliased 6-display-pixel direction segments, skips invalid image pixels and invalid/non-finite decoded directions for arrow placement only, writes the augmented variants as one natural-size horizontal `dir_vis.jpg` strip with a single top label band, and writes sample/checkpoint/per-augmentation drawn-count metadata to `dir_vis_summary.txt`. The valid mask gates model normalization and arrow placement, but does not black out display pixels.
- `--dir-vis --dbg-dirs` adds a second row to `dir_vis.jpg`. Column 1 is the raw unaugmented patch without direction arrows. The remaining columns copy the unaugmented center crop whose side is half the center-patch image side into the center of each transformed patch, run inference on those pasted variants, and render their direction overlays.
- V0.1 patch line-tracing inspection is exported with `--line-trace-vis --checkpoint <snapshot> --export-dir <dir>`.
- Line-tracing inspection uses the same deterministic `--sample-index` ordering as training, prefetch, and augment-vis, loads the center side-strip patch, runs the checkpointed direction model, decodes the Lasagna ambiguous two-cos-channel output, and traces from the transformed CP in both directions.
- The line tracer bilinearly samples the decoded per-pixel direction field, flips sampled directions as needed to maintain forward/backward sign continuity, and steps in strip-pixel coordinates.
- The line tracer stops when the next point would enter the configured receptive-field border margin, when the sampled direction is invalid, or when image validity around the bilinear sample is insufficient. By default the receptive-field margin is `model_depth`; `--line-trace-rf-margin` can override it for inspection.
- The default line-trace step is `4.0` strip-image pixels and can be overridden with `--line-trace-step`.
- Line-tracing inspection writes `line_trace_vis.jpg` as a two-column image by default: the first column is the original transformed strip line plus the unaugmented direction-traced line, and the second column is the same original patch with a flock of traces from random combined geometric test-time augmentations mapped back through TTA output-to-reference coordinate grids.
- Line-trace test-time augmentations are deterministic per `sample_index` and are sampled from the regular training geometric augmentation ranges: shift, rotation, shear, scale, smooth offset, and flips. Value-only augmentations are not applied for line tracing.
- `--line-trace-tta-count` controls the number of random geometric TTA variants and defaults to `100`. `--med-tta-count` is accepted as a compatibility alias for the same count.
- Line-trace TTA constructs augmented coordinate grids from the base patch coordinates, samples the volume at those coordinates, runs the model and tracer in augmented patch coordinates, then uses the TTA output-to-reference coordinate grid to map traced points back into original patch coordinates before drawing. It writes per-TTA trace counts to `line_trace_summary.txt`.
- `--line-trace-vis --med-tta` adds a third `line_trace_vis.jpg` column for
  median test-time augmentation tracing. Median TTA traces in the unaugmented
  reference patch space; at each trace step it transforms the current reference
  point into the reference/TTA direction fields, samples decoded Lasagna
  ambiguous directions there, transforms orientations back to reference space,
  keeps only the ambiguous sign aligned within 90 degrees of the previous
  reference-space step direction, then takes and normalizes the component-wise
  median direction before stepping. The median trace uses the same random TTA
  field list as the flock column. `line_trace_summary.txt` records
  `med_tta=true`, `line_trace_tta_count`, and the median trace point count.
- Trace-to-control-point inspection is exported with `--trace2cp-vis
  --checkpoint <snapshot> --export-dir <dir>`.
- Trace2CP inspection uses the same deterministic `--sample-index` ordering as
  training, prefetch, augment-vis, dir-vis, and line-trace-vis. The sampled
  control point is the start CP. The default target is the next control point
  in the same fiber; `--trace2cp-target-offset` changes the relative target and
  `--trace2cp-target-cp-index` selects an absolute target CP index in the same
  fiber. The target CP must be in range and different from the start CP.
- `--trace2cp-vis --fiber-json <path>` runs whole-fiber Trace2CP visualization
  for an explicit fiber JSON. In this mode the runner narrows a single-dataset
  config to `fiber_paths=[<path>]` before constructing the loader, so it loads
  only that fiber while reusing the configured Lasagna manifest, volume scale,
  cache, and sampler context. It must not require the fiber to match the
  configured dataset glob/list, and it must not introduce a separate
  manifest-less fiber loading path. Whole-fiber mode uses all in-range CP pairs
  for the non-zero `--trace2cp-target-offset`; the default offset `1` evaluates
  adjacent pairs `(0,1), (1,2), ...`. It cannot be combined with
  `--trace2cp-target-cp-index`.
- Whole-fiber Trace2CP must continue past CP pairs whose segment cannot be
  constructed or traced because of invalid local data such as zero
  Lasagna `grad_mag` normal samples. Skipped pairs are reported to stdout and
  listed in `trace2cp_fiber_summary.txt`; the command fails only if every
  requested pair is skipped.
- Trace2CP loading constructs a side-strip segment that spans the start and
  target CPs plus receptive-field/visualization margin. The segment strip
  height is eight times the configured patch height so traces have more vertical
  room before entering the RF margin. It uses the same
  Lasagna manifest normal sampling and VC3D-equivalent side-strip coordinate
  construction as CP-local patches, but anchors the start CP at an explicit
  strip x-coordinate so the target CP lies in the same image at its arc-length
  column. It does not use the neural-tracing 3D crop loader.
- Trace2CP samples the center strip-z image only, runs the checkpointed
  direction model, decodes the Lasagna ambiguous two-cos-channel output, and
  traces the same selected segment in both directions: start CP to target CP,
  and target CP back to start CP on the same segment strip. Each directional
  trace uses `--line-trace-step` and the line-trace receptive-field margin
  default `model_depth`, overrideable with `--line-trace-rf-margin`.
- Each Trace2CP direction initializes its ambiguous-direction sign from the
  vector pointing from that direction's start CP to its target CP. The reverse
  trace therefore starts at the second CP and is seeded toward the first CP.
- Per-direction Trace2CP endpoint diagnostics still evaluate the traced y
  coordinate at that direction's target CP x-column. If the trace crosses that
  x-column, the y coordinate is linearly interpolated between bracketing trace
  points. These endpoint scores remain in the summary as diagnostics, but they
  are not the public `trace2cp_score`.
- The public Trace2CP metric is `trace2cp_error`: the mean target-column y
  error divided by the horizontal start-to-target CP span. The forward trace is
  linearly interpolated where it reaches the target CP x-column and compared
  to the target CP y. The reverse trace is linearly interpolated where it
  reaches the start CP x-column and compared to the start CP y. The two raw y
  errors are averaged before division by horizontal span.
- Target-directed Trace2CP traces must normally stop by reaching the opposite
  CP x-column. When a step crosses that column, the returned trace must append
  an exact linearly interpolated point at the target column before terminating
  with reason `target_column`.
- If a trace explicitly stops before the opposite CP x-column because it hits
  the RF margin, invalid sampled data, or an invalid predicted direction, that
  direction uses the default maximum y error for the segment: vertical distance
  from the CP centerline y to the nearest usable vertical strip edge after
  RF-margin exclusion. The same maximum y error caps pathological endpoint y
  errors. This intentionally treats exact early/late edge intersection as noise
  for now.
- If a target-directed Trace2CP trace terminates by exhausting `max_steps`, that
  is an internal budget failure and must raise a visible error. It must not be
  scored through the missing-target-column maximum-y fallback, because that can
  hide traces that stop far before the opposite CP column.
- The previous center-biased closest-approach value remains available only as a
  refinement/visualization diagnostic named `refine_score`. It must not be used
  as the public Trace2CP metric or as the training best-checkpoint criterion.
- Trace2CP builds a CP-to-CP initialized segment from the two closest-approach
  partial traces. Each CP stays fixed. Each partial trace is corrected only in
  y, with zero correction at its CP and linearly increasing correction toward
  the closest x position. At the closest x position both traces are warped to
  the midpoint between their original y values. The two warped partial traces
  are fused and resampled by arc length using `--line-trace-step`.
- Trace2CP also runs a small deterministic refinement of the fused line that
  reduces local direction mismatch against the sampled direction field while
  discouraging uneven segment spacing. The refined line keeps the two CP
  endpoints fixed.
- `--trace2cp-refine-iterations N` enables iterative Trace2CP refinement after
  the initial pass. Iteration `0` is the normal Trace2CP evaluation. Each extra
  iteration smooths the previous selected fused CP-to-CP trace, keeps the CP
  endpoints and x columns fixed during smoothing, converts the smoothed
  patch-space `(x,y,z)` trace back through the previous segment source to
  volume coordinates, builds a fresh side-strip segment from that volume-space
  curve, and reruns the same Trace2CP scoring mode. `N=0` preserves the current
  single-pass behavior.
- Iterative Trace2CP refinement must resample the volume from the refined
  curve geometry. It must not geometrically warp, bend, rotate, or otherwise
  reuse the previous strip image as the next pass input.
- A refined pass must be equivalent to running Trace2CP on an independent
  line source: after converting the previous fused trace to volume-space line
  points, the loader must build a fresh segment source with endpoint context
  before the start CP and after the target CP. Both forward and reverse traces
  must therefore have the same valid local neighborhood at their start points
  as they do for an original fiber-json line.
- `--trace2cp-refine-smooth-window` controls the finite Gaussian smoothing
  window used between iterations and defaults to `5`. Even values are rounded
  up to the next odd window. The smoothing keeps x columns and both CP
  endpoints fixed.
- Single-pair refinement outputs keep the initial pass as `trace2cp_vis.jpg`
  and `trace2cp_summary.txt`. Extra passes write `trace2cp_vis_it1.jpg`,
  `trace2cp_summary_it1.txt`, then `it2`, etc. If z-layer TIFF export is also
  enabled, extra passes write `trace2cp_z_layers_it1.tif`, etc.
- Trace2CP CLI runs print a compact stage timing table after the metric/debug
  summary. Single-pair mode prints `trace2cp timings`; whole-fiber mode prints
  `trace2cp fiber timings` aggregated across valid pairs. Rows are grouped by
  stage and include count, total milliseconds, mean milliseconds, and max
  milliseconds, covering inference, source sampling, tracing, debug rendering,
  and file output stages where applicable.
- Slow Trace2CP dynamic-programming solves print live progress before the final
  timing table. DP progress is opt-in at CLI call sites and uses rows
  `trace2cp dp start`, `trace2cp dp progress`, `trace2cp dp done`, or
  `trace2cp dp failed`. Progress rows include the DP label, solved columns,
  elapsed seconds, and `eta_s` on progress rows. The low-level DP helper is
  quiet by default for unit tests and internal direct calls.
- Trace2CP CLI side, side-z, and top-model DP solves use a torch-vectorized
  backend on the active model device. The backend keeps the DP column
  recurrence sequential but vectorizes per-column work across z layers, rows,
  sampled transition columns, and move chunks. The existing NumPy/Python DP
  remains the fallback when no torch device is supplied.
- Side-z DP must not infer or optimize unreachable z layers. Since the path is
  anchored at the center layer at both CP columns and transitions can move only
  one z layer per DP column, the effective layer bound is capped by the number
  of horizontal transitions. The side DP vertical move lattice may use a broad
  compute search band, but that band must be independent of the configured
  candidate-angle limit.
- Trace2CP uses `--med-tta` to determine whether TTA is used. Without
  `--med-tta`, it traces and scores both directions on the base strip
  direction field. With `--med-tta`, it builds deterministic random geometric
  TTA direction fields using `--line-trace-tta-count`, default `100`, and
  traces both median-TTA directions in the reference segment strip.
- Trace2CP supports an optional inspection/refinement mode enabled by
  `--trace2cp-combined` or `--trace2cp-use-presence`. In this mode the selected
  trace uses the regular stepwise candidate-fan tracer by default, scoring side
  direction at both the current/last point and the candidate point, plus
  optional presence. The non-combined reference tracer remains the public
  target-column `trace2cp_error` path. The monotone-x dynamic-programming
  backend is experimental and must only run when `--trace2cp-dp` is explicitly
  supplied.
- The side-strip DP state is `(side_z_layer, y, prev_dy, prev_dz)`. It uses
  fixed 4 px horizontal transitions, plus the exact target column, and
  integrates angle-space direction alignment cost across every crossed pixel
  column. The sampled alignment is frame-ambiguous:
  `theta = degrees(acos(abs(dot(path_tangent, layer_direction))))`, and the
  cost is
  `(theta / 10)^2 * (1 + max(theta - knee, 0) / knee)` before applying
  `direction_weight`. Transition samples use fractional bilinear
  interpolation in strip row and z-layer coordinates, not rounded nearest
  lookup. Because decoded Lasagna directions are sign-ambiguous, all
  interpolated direction-vector corners are sign-aligned to the candidate
  transition tangent before blending. Invalid or missing direction pixels add a
  fixed penalty instead of breaking the path. Side-strip DP does not apply a
  default per-step z movement penalty; its default z regularization is
  second-order dz smoothness, currently `0.5 * (dz_current - dz_previous)^2`,
  so steady z motion is allowed while abrupt z-step changes are discouraged.
- The side-strip DP still uses `--line-trace-step` only for resampling the
  selected fused output trace and for the public trace visualization density.
  It must not use `--line-trace-step` as the DP transition length.
- The side-strip DP uses the existing Trace2CP candidate-angle setting as the
  local angle-excess knee in that penalty. With the default 25 degree knee,
  10 degrees costs roughly 1, 20 degrees roughly 4, and 45 degrees roughly
  36. This setting must not cap global horizontal slope or vertical moves,
  because valid local fiber directions can be steeper than 45 degrees.
- The default side-strip DP dy smoothness penalty is zero. The default
  side-strip DP dz smoothness penalty is nonzero as described above and should
  discourage lateral/z jitter without penalizing total z travel.
- `--trace2cp-combined-mode direction` is the only active combined mode.
  `--trace2cp-combined-mode embedding`, `--trace2cp-use-embedding`,
  `--trace2cp-combined-mode image`, and `--trace2cp-use-image` are removed from
  the active tracer and must fail clearly if requested. Legacy helper
  functions may remain as inactive implementation experiments, but runner
  Trace2CP selection must not route through embedding or image similarity.
- `--trace2cp-use-presence` adds an orthogonal sheet/fiber-presence score to
  the active combined tracer. It samples the sigmoid presence probability from
  the same selected layer as the direction field and adds
  `trace2cp_combined_presence_weight * (1 - presence_probability)` to the
  candidate/transition cost. This requires a checkpoint/model output with a
  presence channel and fails clearly if the channel is absent. Visualization appends
  fixed-scale presence debug output when presence scoring is active: single-pair
  `trace2cp_vis.jpg` gets a presence column, whole-fiber
  `trace2cp_fiber_vis.jpg` gets a presence row, `0` renders black, `1` renders
  white, invalid pixels are black, and the fiber line, CPs, and selected traces
  are overlaid. When z-search is enabled, the z debug visualization must also
  show forward, reverse, and fused z-corrected presence maps selected
  column-by-column from the same trace z layers as the z-corrected image.
  Whole-fiber presence visualization must use the fused z-corrected presence
  when it is available.
- Trace2CP z-search uses raw per-layer side-presence by default. Adding
  `--trace2cp-presence-blur` makes Trace2CP use a cache-level
  Gaussian-smoothed side-presence view for presence scoring and presence
  display. The smoothing is weighted by valid pixels and runs over side-z plus
  side-image x/y. The side-z pass uses radius 21. The side-image pass uses a
  per-pixel anisotropic Gaussian rotated around the side-z axis to align with
  the local predicted side direction: radius 5 along the direction and radius 1
  across it. The kernel is symmetric, so direction sign ambiguity does not
  change the result. Non-z Trace2CP presence scoring remains unblurred because
  there is no side-z stack.
- Trace2CP visualization also appends VC3D-style top-strip output sampled from
  volume coordinates, not warped from rendered side-strip pixels. Single-pair
  `trace2cp_vis.jpg` gets a top-strip debug column. Whole-fiber
  `trace2cp_fiber_vis.jpg` gets top-strip rows stitched into the same global CP
  x-coordinate system as the side-strip rows. The original/init comparison top
  strip uses the same pair-local line window and Lasagna/VC3D frame
  construction as the side-strip segment, but rows are offset along
  `frame.side` as in VC3D `lineSurface`. Visualizations must also include a
  traced fused top strip projected to the central z slice: for each output
  column, interpolate the fused trace, sample the segment coordinate grid and
  Lasagna row-normal axis at that traced side-strip point, derive the
  top-strip side axis from traced tangent and row normal, then sample rows
  along that side axis with zero side-z offset. When z-search is active, the
  visualization additionally appends a traced fused z-corrected top strip using
  the fused trace's selected-scale `z_voxels` value as an out-of-plane side-z
  offset before the top-strip side-axis offset. This is visualization-only and
  must not change Trace2CP scoring.
- When z-search is active and the side model exposes a sheet/fiber-presence
  head, Trace2CP top-strip visualization also appends fixed-scale side-presence
  z-pillar rows below the regular top-strip slices. For each output column `x`,
  each pillar row samples one inferred side-slice layer at `(x, trace_y(x))`;
  the image height is `2 * trace2cp_z_max_layer + 1`, so `+/-40` produces an
  81 px tall z-pillar image. Separate z-pillar panels may be shown for the
  original/init trace, the traced fused central-z line, and the z-search fused
  line. For the z-search fused line, each column is shifted by that column's
  selected z value (`round(z_voxels / z_step_voxels)`), so the center row
  represents relative z=0 at the layer actually used by the trace. These rows
  are side-z-stack projections rather than true top-strip surface predictions;
  if the side presence field is broad or similar across shifted layers they
  can resemble a narrow side-presence slice. They are visualization-only, do
  not use the optional top-view model, and must not affect Trace2CP scoring,
  z-search, or training.
- `--trace2cp-top-model-dir-vis` requires a checkpoint with
  `top_model_state_dict`. It samples a fixed top-strip normal-offset stack
  around the traced fused top strip using offsets `-4..+4` selected-scale
  voxels in one-voxel steps, runs the jointly trained top-view model on every
  layer, and appends sparse direction indicators from an aligned median
  direction field. Per pixel, only valid layer directions within 45 degrees of
  image-horizontal are considered; each Lasagna-ambiguous direction is
  normalized and sign-aligned before taking the median so opposite signs cannot
  cancel. If a z-corrected fused trace is available, that trace is used as the
  stack center; otherwise the central-z fused trace is used. The same fused
  top-direction field is traced from each CP along the top-strip center row
  until the opposite CP x-column, invalid direction, edge, or max-step guard,
  and those two traces are drawn with equal visual weight on the debug panel.
  The panel also draws a monotone-x dynamic-programming path connecting the two
  CP columns on the top-strip center row. That DP path's state is
  `(top_offset_layer, y, prev_dy, prev_dz)`, so it may transition between
  neighboring top-offset layers with a fixed z-transition penalty of
  `0.1 * abs(delta_layer)` while also preferring smooth step sequences. The
  default second-order penalties are zero; the first transition has no
  smoothing cost because no previous step exists. There is no default
  absolute-y row penalty because that would bias the path toward a row rather
  than smoothing its slope. It uses fixed 8 px horizontal transitions, plus the
  exact target column, and integrates direction alignment
  cost `1 - abs(dot(path_tangent, layer_direction))` across every pixel column
  crossed by each transition, using fractional row/z interpolation from the
  direction field. The vertical transition band scales with the horizontal
  step, and start/target rows and layers are exact at the CPs. Invalid or
  missing direction pixels in the selected layer add a fixed penalty instead of
  blocking the path, so the diagnostic path still connects the CPs while
  preferring valid pixels where available.
  The visualization also appends optimized-line diagnostics derived from that
  DP path: a top strip resliced around the DP top-row path and selected
  top-offset layers, a side slice reconstructed column-wise from the same
  optimized side displacement, and matching top z-pillar plus side-column
  presence panels when side-model presence is available. The optimized side
  displacement is the sum of the selected top-offset layer and the DP
  top-row offset from the old center row. In the optimized top-strip panel,
  the optimized path is the slice center and is drawn as a straight centerline,
  not as the pre-reslice curved path. Side-slice and presence diagnostics must
  build a visualization z-plane cache whose bounds cover this combined
  optimized side displacement; using only the raw selected top-offset layer
  range can incorrectly turn out-of-cache columns black. These panels use the
  optimized line only for visualization and do not feed back into Trace2CP
  scoring.
- Z-corrected side-image and side-presence visualization helpers must infer
  requested cache layers on demand with the z-plane cache API. They must not
  treat `plane_cache.layers` as a complete layer set, because visualization
  caches may start with only the center layer populated.
  During top trace integration, ambiguous direction signs must be resolved
  before bilinear interpolation by flipping each of the four neighboring pixel
  direction samples, if needed, so it agrees with the current trace direction;
  otherwise opposite signs from the Lasagna two-cos encoding can cancel or
  flip the sampled direction. This is visualization-only and must not change
  Trace2CP scoring or z-search layer selection.
- `--trace2cp-side-top-z-experiment` is an opt-in single-pair diagnostic. It
  is exclusive: when set, the runner writes only the side/top-z experiment
  artifacts and does not run the normal Trace2CP overlay/refinement chain,
  public `trace2cp_error` export, training metric, or best-checkpoint
  selection. The
  experiment runs regular stepwise side-strip traces from both CPs while also
  carrying a selected-scale z/offset state. Side x/y stepping must use the same
  candidate fan scoring semantics as the normal forward/backward combined
  tracer: interpolate side direction at the current/last point and at each side
  candidate point from the side prediction for the current z layer, using
  ambiguity-aware two-cos direction interpolation, and include optional side
  presence scoring when `--trace2cp-use-presence` is active. It must not score
  embedding/image similarity or run DP in this diagnostic. Top
  inference must not run for all side candidates. After the side candidate is
  selected, the experiment builds one local top patch centered at that accepted
  side point and runs the checkpoint's top-view model on that patch to update
  only the carried z/offset state. The local top patch x axis is derived from the
  sampled side-view direction: the side-strip tangent is tilted within the side
  tangent/normal plane according to the side direction. The top patch keeps the
  side-strip lateral axis as the second in-plane axis, so this experiment only
  corrects angle relative to the side-view normal and does not optimize roll or
  arbitrary rotation around the fiber line. The top direction used for the
  offset update is an ambiguity-aligned weighted median over a normal
  neighborhood, default radius 20 px. The experiment writes separate
  `trace2cp_side_top_z_experiment.jpg` and
  `trace2cp_side_top_z_summary.txt` artifacts. The JPG must stay compact and
  diagnostic-specific: forward side trace with z-corrected image, backward side
  trace with z-corrected image, forward z-corrected presence, backward
  z-corrected presence, original top strip, forward traced top strip with
  z-correction, and backward traced top strip with z-correction. It must not
  draw per-step top-direction ticks there or reuse the full Trace2CP overlay
  rows for
  fused/reference/similarity/DP debug.
  The experiment additionally writes every local top slice actually used for
  z-update inference to `trace2cp_side_top_z_top_slices/` and a matching
  native-resolution direction overlay to `trace2cp_side_top_z_top_overlays/`;
  filenames are prefixed `fw_` or `bw_` by trace direction. These generated
  directories clear stale JPGs before each export. XYZ trace positions and z
  offsets are subpixel/floating-point throughout stepping; rounding is limited
  to side z-layer prediction lookup and column-wise display reconstruction.
  Because this diagnostic repeatedly samples local top patches and runs top
  model inference, it prints throttled `trace2cp side_top_z progress` rows for
  the forward and backward traces. Each row includes a small progress bar,
  accepted steps versus expected horizontal steps, top-patch and invalid counts,
  current z offset, elapsed time, ETA, and the final termination reason.
- Combined Trace2CP is an inspection/refinement path. It does not replace the
  public `trace2cp_error` definition, the direction-only tracer, training loss,
  or best-checkpoint selection unless explicitly enabled by the command-line
  flag. `--med-tta` is supported only by the stepwise combined tracer, not by
  the explicit `--trace2cp-dp` backend.
- `--trace2cp-vis --trace2cp-combined --trace2cp-z-search` enables an
  experimental side-strip z-search mode. It requires combined tracing and
  cannot be combined with `--med-tta`. By default this is the regular stepwise
  candidate-fan z-search that existed before the DP experiment: each accepted
  side step may choose the current or neighboring z layer. Existing Trace2CP
  commands without `--trace2cp-z-search` keep the center strip-z image-only
  behavior. Adding `--trace2cp-dp` switches this z-search to the experimental
  monotone DP backend.
- Trace2CP z-search derives additional segment-strip planes from one accepted
  center segment source. The center source is built once from the CP-to-CP
  line window and Lasagna normals, including the row-axis sign alignment used
  for whole-fiber Trace2CP. Side-strip axes are explicit: image x follows the
  fiber tangent/arc direction, image y follows the Lasagna mesh-normal row
  axis, and z-search layers move along the remaining out-of-plane side axis
  aligned with the VC3D frame side direction, approximately
  `mesh_normal x tangent`. State layer `k` represents
  `z_voxels = k * --trace2cp-z-step-voxels` along that axis. Volume/model
  inference must run at no finer than one selected-scale voxel spacing: when
  `--trace2cp-z-step-voxels >= 1.0`, the requested state layer is sampled
  directly by adding `side_axis_zyx[y,x] * (z_voxels * volume_spacing_base)`
  to every center coordinate before volume sampling; when
  `--trace2cp-z-step-voxels < 1.0`, only the bracketing integer
  selected-scale side-z voxel offsets are sampled and inferred. Direction and
  sheet/fiber-presence fields for sub-voxel state layers are interpolated from
  those integer layers, with ambiguous direction vectors sign-aligned before
  interpolation and normalized afterward. It must not use the side-strip
  image-y/row axis, a global normal, a row-coordinate approximation, an
  image-space shift, or an unrelated rebuilt plane. The default
  `--trace2cp-z-step-voxels 1.0` means layer `k` is offset by `k`
  selected-scale voxels along the segment strip side-z axis.
  `--trace2cp-z-max-layer` bounds lazy expansion and defaults to `4`.
- Default z-search lazily samples side-strip layers as the stepwise candidate
  tracer requests the current and neighboring z layers. Inference is
  deterministic and stores each layer's sampled image, valid mask, decoded
  direction field, and optional presence field. For sub-voxel z steps, lazy
  sampling stores both requested state layers and the integer inferred layers
  used to build them. Each selected path point carries `x`, `y`, and
  selected-scale `z_voxels`; direction and presence costs are sampled from the
  selected state layer, which may be interpolated from neighboring inferred
  integer side-z layers. If presence is used, those state-layer presence fields
  are first smoothed over side-z and strip x by the cache-level presence blur.
- With explicit `--trace2cp-dp`, z-search infers the bounded reachable layer
  stack for the pair before running the side DP. The DP may transition between
  neighboring z layers without an absolute z movement penalty, while its
  default dz smoothness term discourages abrupt z-step changes. DP output is
  already a fused CP-to-CP path, so the optimized visualization row uses that
  joint path directly.
- Z-search does not change the public `trace2cp_error`, training test metric,
  or best-checkpoint selection. Those remain target-column y error per
  horizontal CP span.
- Single-pair z-search visualization adds a z-corrected column. It contains
  separate forward and reverse views because each trace direction can choose a
  different z layer per x column. It also contains a fused z-corrected view and
  a fused z-layer map row so the selected layer per output column is visible
  even when neighboring sampled planes look similar. Each z-corrected image is
  assembled column-by-column by rounding the trace/fused z value to the nearest
  z-search state layer and copying that state's sampled image column. For
  sub-voxel z steps, interpolated states reuse the nearest integer-inferred
  side-z image. It must not re-sample the volume and must not interpolate image
  values between z layers; columns without a trace/fused z value and columns
  whose rounded layer is missing render black and are counted in summary/debug
  output.
- `--trace2cp-vis --trace2cp-z-search --trace2cp-z-layers-tif` exports the
  already inferred z-search layer cache as TIFF debug stacks. Single-pair mode
  writes `trace2cp_z_layers.tif`; whole-fiber mode writes one pair-local TIFF
  per valid pair under `trace2cp_z_layers/` because segment strips can have
  different shapes. Pages are uint8 and non-interleaved: all sampled slice
  images first in sorted inferred z-layer order, then all available
  sheet/fiber-presence maps in the same sorted inferred z-layer order. For
  sub-voxel z steps, the TIFF stack exports only the actually inferred integer
  side-z layers, not every interpolated DP/search state. The export must use
  the existing z-search cache, must not re-sample the volume, and must not
  interpolate image values between z layers.
- `--trace2cp-vis --trace2cp-obj` is an opt-in single-pair diagnostic export.
  It writes vertex-colored OBJ meshes under `trace2cp_obj/` plus a manifest.
  OBJ geometry must come from the same sampled Trace2CP coordinate grids used
  for image loading: center side strip, z-search selected side-strip columns,
  original top strip, traced fused top strip, and z-corrected traced top strip
  when those surfaces exist. Vertex colors are grayscale scalar values from the
  corresponding volume image (`0..255`) or side-model sheet/fiber presence
  (`0..1` for raw presence, `0..255` for z-corrected debug presence). Quad
  faces are emitted only where all four vertices are valid. The flag is not
  currently supported by whole-fiber `--fiber-json` Trace2CP output.
- Single-pair `trace2cp_vis.jpg` includes an additional embedding-debug column
  when the checkpoint exposes embedding channels. The column renders cosine
  similarity maps for the start CP embedding, target CP embedding, same-fiber
  CP-bank mean similarity when the combined Trace2CP bank is available,
  forward trace-progress last-point columns, and reverse trace-progress
  last-point columns. For the forward/reverse panels, each newly placed trace
  point paints the vertical column band around itself using the previous
  accepted trace point's embedding as the similarity reference; the band radius
  is `ceil(step_px / 2)`, and small overwrites are allowed.
  These maps are fixed-scale cosine displays (`-1..1` mapped to
  `0..255`) and are visualization-only; they must not affect tracing,
  refinement, metrics, or best-checkpoint selection.
- Trace2CP TTA samples from the regular training geometric augmentation ranges
  but forces y-shift to zero and scale to one for long-strip target-column
  semantics. Each TTA field is built by transforming the segment coordinate
  grid first, then sampling the volume at those coordinates. It must not warp
  the already sampled base segment image.
- Trace2CP TTA output canvases are sized so the transformed base segment-strip
  corner footprint fits in the TTA image. Pixels that map outside the base
  coordinate strip or volume stay invalid/black.
- The Trace2CP median trace is stepped in the reference segment strip by
  sampling the reference and TTA direction fields, mapping each current
  reference trace point into each TTA field through the prebuilt
  reference-to-TTA map, mapping TTA directions back to reference coordinates
  through each TTA output-to-reference coordinate grid, resolving ambiguous
  signs against the previous step, and using the normalized component-wise
  median direction. It must not locate reference points in TTA fields by
  scanning the dense output-to-reference grid.
- `--trace2cp-vis --med-tta --vis-tta` writes `trace2cp_tta/reference.jpg`,
  one `trace2cp_tta/random_NNN.jpg` per generated TTA field, and a contact
  sheet. Each TTA debug image shows the sampled TTA slice with the transformed
  base-strip corner outline and start/target CP markers.
- Trace2CP writes `trace2cp_vis.jpg`, writes `trace2cp_summary.txt`, and prints
  a dedicated public-metric stdout line beginning with `trace2cp_error=...`.
  Additional stdout lines are diagnostics and must not duplicate the selected
  public metric label. The summary includes sample index, fiber path,
  start/target CP indices, trace mode, public `trace2cp_error`,
  target-column metric raw y error in pixels, horizontal CP span, refinement
  diagnostic score, endpoint diagnostic scores, per-direction raw errors,
  target x-columns, reach statuses, termination reasons, and trace point
  counts. The JPG is a labeled vertical stack with rows for full bidirectional
  traces, partial traces up to the closest point, the fused CP-to-CP line, and
  the optimized refinement. Without `--med-tta`, this stack is the
  reference-only inference result. With `--med-tta`, the JPG has two columns:
  the selected median-TTA result first, and a second reference-only inference
  column using the base direction field without TTA. It does not draw score
  text over image pixels.
- With `--trace2cp-refine-iterations`, the base `trace2cp_vis.jpg` remains the
  initial pass for compatibility; extra pass visualizations use the `itN`
  suffix. Each `itN` pass uses the same drawing structure and public
  `trace2cp_error` reporting semantics as the initial pass.
- Whole-fiber Trace2CP mode writes `trace2cp_fiber_vis.jpg` and
  `trace2cp_fiber_summary.txt`, and `trace2cp_fiber_debug.txt`. Each CP pair
  is loaded, traced, and measured with the same pair-local Trace2CP path as the
  single-pair command. The final
  visualization is composed afterward by mapping each pair-local segment image,
  centerline, CP markers, selected traces, and optimized line into a shared
  arc-length x coordinate system for the selected fiber. The mapping uses each
  pair's local start/target CP image columns and the corresponding global
  start/target CP arc-length columns. Pair-local y orientation is fixed before
  image sampling by shared-CP row-axis alignment, not by guessing after
  composition. The debug file and stdout include per-pair start/target CP strip
  coordinates, strip-space CP deltas, start/target row axes, frame vectors, and
  3D CP deltas projected into the start frame. The image layer uses dense rectangular valid-mask averaging of
  the already sampled segment images; it must not use sparse per-pixel
  splatting that can introduce display holes. Metric errors and traces are
  still computed pair by pair. The JPG uses the same four-row Trace2CP structure as
  single-pair output: full bidirectional traces, partial closest-approach
  traces, fused CP-to-CP line, and optimized CP-to-CP line. Skipped-pair counts
  and reasons are included in the summary. Whole-fiber metric output is the
  average public `trace2cp_error` over all valid CP-pair segments and is
  printed on its own stdout line as `trace2cp_error_mean=<value>`.
- With `--trace2cp-refine-iterations`, whole-fiber mode writes additional
  aggregate iteration images and summaries as `trace2cp_fiber_vis_it1.jpg` and
  `trace2cp_fiber_summary_it1.txt`, then `it2`, etc. The unsuffixed whole-fiber
  outputs remain the initial pass.
- Trace2CP target-column crossing takes precedence over RF-margin rejection for
  the next step in each direction. If a step crosses that direction's target
  x-column and would also enter the RF margin, the trace is considered to have
  reached the target column, an exact interpolated target-column point is
  appended to the trace, and the score is computed at that point. RF-margin
  stop reasons should identify whether the x margin, y margin, or both were
  hit. `max_steps` exhaustion is not a valid scored stop reason for
  target-directed Trace2CP traces and must raise instead.
- Tests use fake/local arrays and monkeypatched readers where possible and must not require network access.
- `docs/code_structure.md` documents the current implemented module structure, data flow, config shape, runner outputs, and local workflow caveats; `planning/specs.md` remains the normative behavior source.
- Future changes that affect public config, data flow, sampling, caching, augmentation, runner outputs, tests, or local workflow must update both the relevant specs and code docs.
