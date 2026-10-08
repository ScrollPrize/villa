# Paired long-range embedding training

Enable `embedding.paired_patches: true` with `local_contrastive` and
`scroll_balanced_arclength`. Existing behavior is unchanged when omitted.

One pair contains two independently positioned patches on the same full fiber.
Existing scroll/group/arclength weighting chooses the first location/fiber;
the second draws a fresh arclength location on that fiber. Pair positions can
overlap by chance; no minimum distance is imposed, so short fibers remain usable.

Brightness, contrast, gamma, noise strength/seed, blur sigmas and blur activation
are shared. Affine, center shift and smooth deformation are independent.
Directional blur remains fiber-aligned in each independently transformed patch.
Shared noise seed means the same noise realization in output-grid coordinates.
Intensity normalization still uses each actual patch's data, with identical policy.

Each patch supplies anchors. Positives choose own/partner patch with probability
one half and sample GT arclength there. Negatives sample both patches jointly,
proportional to detached presence after each patch's own normal-ellipse exclusion.
The original per-patch counts remain:65536 negatives plus2048 positives at128 cube.
128 patches per update therefore means64 pairs; both outputs receive gradients
in one backward. No cross-rank/microbatch prediction dependencies are introduced.

Existing shared queue/slots, native CT/normal sampling and GPU value augmentation
are reused. Pair metadata is checked before loss/count evaluation. Invalid-normal
replacement stays on the same fiber with unchanged value seed;128 unsuccessful
retries produce an error instead of silently mixing fibers.

Cooperative allocations partition whole pairs; minimum microbatch is2 patches.
Automatic startup calibration measures2,4,... patches through real training.
Pausing still unloads GPU workers normally. Historical non-embedding metrics retain
their old sampling; ordinary full-population tests now load a pair per original
sample slot (twice the patches). Explicit test counts and offsets must be even.

Config: `fiber_trace_3d/configs/train_combined_5scroll_s1a_paired_contrastive_ema_16x8.json`.
LR0.01, combined five-scroll/Paris4 JSON+NML weighting, EMA0.99/0.999 and2M updates
are unchanged. This requested replacement starts from scratch, not a checkpoint.

Approved test launcher: `vesuvius/scripts/run_fiber3d_cooperative_tests` with
`paired`, `paired-data`, `cuda-embedding` suites. GPU suites acquire manager
exclusivity first; the production launch is adaptive through the same manager.

Validation (2026-09-13): 75 CPU tests passed, all configured training/test dataset
entries passed native pair-loading checks, and 7 GPU tests passed including
two-GPU gradient equivalence. Paired training requires bounded accumulation;
the cooperative launcher supplies it automatically.

Start a fresh run with:
```bash
python -m vesuvius.neural_tracing.fiber_trace_3d.train vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_combined_5scroll_s1a_paired_contrastive_ema_16x8.json
```

## Independent short- and long-range banks

`embedding.short_range_channels` optionally divides one embedding output into two
independently normalized cosine spaces. It requires paired local-contrastive
training and must be smaller than `embedding.channels`. Channels before the split
are supervised only within a patch; channels after it use only cross-patch
positives from the same fiber. Presence-weighted negatives follow the same range:
the short bank samples its own patch and the long bank samples both members of the
pair. The configured negative-pair budget is divided equally between the banks,
and training averages their independently normalized mean losses.

The frozen-refinement production config uses 16 channels split8/8:
`train_5scroll_json_eligible_fullres_frozen_refined_split_embedding_ema_16x8.json`.
It starts model, optimizer, EMA and update count from scratch. It does not resume
the refinement model checkpoint; only the immutable refined training geometry is
imported from the checkpoint's SHA-verified archive. Tests continue to use the
original fixed GT because the saved test-patch identities predate paired sampling.

Inference consumers must pass `embedding_bank="short"` or `"long"` for a split
checkpoint. Concatenating the banks is rejected because their cosine spaces have
different supervision. Legacy embedding checkpoints and configs are unchanged.
