# Reviewed fiber normal, width and H/V training

Top-level configuration enables any subset of the new heads:

```json
"annotations": {
  "normal_weight": 10.0,
  "width_weight": 0.01,
  "hv_weight": 1.0
}
```

Zero or omitted weight disables that head. These settings are saved in checkpoints.
The existing unconditioned U-Net adds six sign-ambiguous normal channels, one
positive width channel, and one binary H/V channel. Legacy direction/presence
and embedding channel positions do not change. Default forward/inference still
returns legacy fiber products. `model(volume, return_annotations=True)` returns
raw auxiliary logits in `annotations`, ordered normal, width, H/V, skipping
disabled heads. Normal decoding uses sigmoid and the shared Lasagna3x2 decoder;
width uses softplus; H/V uses sigmoid (H=0, V=1). Auxiliary volume export is not
added by this training task. Conditioned-decoder models reject these heads.

## Data and geometry

Prepare data using `vesuvius/scripts/collect_reviewed_fibers.py`; see its adjacent
Markdown documentation. Parent H/V is inherited, never re-estimated on a piece.
Width-annotated pieces require dense baked normals from the recorded Lasagna
prediction plus shared VC3D CP corrections. No normal mask/remote loader is added
to the training pipeline: compact line targets travel in the existing fast queue.
Gaussian/embedding sampling can also use baked normals, with no runtime Lasagna
manifest. Invalid normals exclude normal/width labels; H/V remains usable.
Normal-dependent Gaussian/embedding patch sampling rejects invalid locations.

Only eligible selected-fiber spans are supervised. Width/HV labels are sampled
uniformly along source arclength at roughly one input voxel spacing. Normal labels
cover the Gaussian tube, like direction; the same rasterizer picks the winning
segment for both at overlaps. No auxiliary background labels are invented.
Dataset affine transforms, eligible-span splits and refinement resampling
preserve label alignment. Augmentation applies the inverse-transpose local geometry
Jacobian to surface normals. Width scales along the transverse direction
`cross(fiber_tangent, surface_normal)`, including smooth geometric deformation.
Widths are **full widths in input voxels**: level1 gives half the annotated base
width before augmentation. H/V is a source classification, unchanged by augmentation.

## Losses and reporting

Normal: mean squared error in six-channel encoding, weighted by projection
magnitudes of the normal axis and Gaussian falloff; divide by six times summed
falloff, exactly like direction (using the normal axis instead of the tangent).
Predictions are sigmoid on the output lattice; no centerline-logit interpolation.
The existing rasterizer retains one int32 owner per voxel only with normal
supervision enabled, then gathers sparse segment normals on Gaussian support.
Width: SmoothL1 (beta=1 input voxel), weighted by Gaussian falloff on the tube
and divided by labelled falloff mass. H/V: binary cross entropy, divided by
valid line-point count over the logical batch. There is no
equal-patch or class balancing; missing labels have zero count. Denominators depend only on GT.
They use the existing global-count
prepass and **one backward per microbatch**, including gradient accumulation.

`train/loss_normal` and corresponding raw/EMA test metrics report **unweighted**
MSE, like `loss_direction`; both receive weight10 in total loss. Width/HV metrics
still report weighted contributions. Sample step units
and existing tests/snapshots remain unchanged. Width magnitude can initially
dominate the total; these smoke-test weights are not convergence-tuned.

## Corrected configuration

Both annotation configs now use all92 baked reviewed pieces from
`/ephemeral/hendrik/fiber3d_train/2026-10-05_reviewed_annotators_124643/train/`.
Validation retains the exact selection/geometry from production launch20261003_222618,
now with baked normals and parent H/V:
`/ephemeral/hendrik/fiber3d_train/2026-10-05_reviewed_failure_union_baked_v1/test/`,32 pieces across
0139,1299,Paris4. Dense tests use the complete original deterministic selection
(test_control_points=0); resident trace tests and cross-section visualization run
at the original100-update intervals. No new holdout is created.

Trace tests are gated on the globally aggregated raw dense-test mean angle first
falling strictly below10 degrees, with nonzero angle support. Once unlocked they
remain eligible even if error rises. The latch survives checkpoint resume and
cooperative pause/reallocation. Dense/raw/EMA and cross-section tests are not gated.

Long config `configs/train_reviewed_annotations_16x8.json`:2M updates, globalbatch128,
target microbatch16, LR0.01/100 warmup, refinement and embedding off; managedGPU0–5.
The short config remains an8-update variant but uses the same restored test protocol.
The runner points at the long config. Started from scratch on manager GPUs0–5:
`reviewed_normal_width_hv_fullres_128_sd1_bs128_ema_20261005_153147`.
Initial raw/EMA tests and first optimizer update verified. The new test copy
resolves the missing-normal-label blocker:71,165 valid
normals,11 PHerc0139 pieces (all H),6 PHerc1299 (all V),15 Paris4 (12H/3V).
One Paris4 piece's previous H classification is corrected to its full parent's V.
Original files remain under `/home/hendrik/data/2026-10-01_reviewed_failure_union_v1`.
Both annotated configs use the copy for dense and resident trace testing; the
trace template remains shared and its split_root is overridden by the config.

After the normal tube/logging correction, resumed current raw at update900 into
`reviewed_normal_width_hv_fullres_128_sd1_bs128_ema_20261005_171209`.
Verified update1000 (128,000 samples), LR0.01, manager GPUs0–5. Pre-correction
normal curves include the10x weight and use different support; do not compare
their displayed scale directly against the corrected curves.

The earlier8-update smoke result and stopped long-run artifacts are historical
only: they used the now-rejected split. That derived dataset has been removed;
all92 original pieces were hash-verified intact. See `annotated_training_audit.md`
for exact changes and requested-versus-agent-selected choices.

## Width in sample overviews

When the width head is enabled, `train_sample_3d/principal_slices` and
`test_sample_3d/principal_slices` append a predicted-width column after the existing
presence columns on all five planes (YX, ZX, ZY, longitudinal and cross-sectional).
The existing loss forward supplies the width logits; softplus gives full width in
network-input voxels. Each sample's `volume_spacing_base` converts to base voxels,
independent of output stride. The fixed grayscale mapping is
`round(clip(width_base, 0, 255))`: one base voxel equals one grayscale unit.
No per-image normalization or presence masking is applied. Existing validity and
CP markers remain. Models without width retain their original overview layout.

Direction-modulated presence views additionally multiply by
`clip(1 - abs(width-center_width)/(0.2*center_width), 0, 1)`.
The same center CP prediction is sampled for all five planes. Raw presence, GT
and width panels are not gated. Invalid/zero reference width gives zero gate.

## Width-aware tube and sideband losses

Gaussian width supervision uses softplus full width in input voxels with SmoothL1,
weighted by Gaussian falloff and divided by labelled falloff mass. It uses the
same winning segment as direction. Source polyline indices and clipping fractions
carry each segment's own widths/normals through the crop halo; the shared annotation
geometry transform applies transverse augmentation stretch at each endpoint.
Widths interpolate along the segment projection. No nearest-row reassignment to
nearby/returning segments is used. A segment with either endpoint unlabelled is unlabelled.
Sideband reference widths use the nearest contributing segment, stable first-win
ties, outside Gaussian support. Sideband-only voxels never enter width tube loss.

Extra left/right sideband negative BCE uses the same linear20% gate, comparing
detached predicted width to transformed GT reference width. Its denominator stays
the original GT sideband area/group count, not the model-dependent gate mass.
Thus dissimilar lobes lose only the extra penalty. Regular positive, distribution
and background presence losses remain unchanged. Width receives no gradient
through this gate; missing labels give zero extra penalty. No-width models retain
their previous sideband behavior. The same code applies to train/raw/EMA tests.

Production weights remain normal10, width0.01, H/V0.01.
