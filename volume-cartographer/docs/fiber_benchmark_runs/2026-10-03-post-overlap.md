# Post-fw/bw piece overlap suppression —1024 pilot

Frozen1879 raw40500 beam8/full-prefix512/iso20 pieces. No early stopping or retracing. Every retained reverse-filter split is independent. Global quality-ranked whole-piece suppression; ties use length then source ordinal. Suppressed pieces cannot suppress others.

Collision: at least50% or80% of the SHORTER piece has matching segments of ONE other piece. Arc-weighted midpoint samples at most4base voxels apart, shared normal ellipse, unoriented angle45°. Presence is length-weighted shared prediction sampling (max valid option), not seed presence. Cost is saved tracer loss/path length, lower wins. No GT is used in selection.

| Variant | Pieces | Retained length | Postprocess s | BP pieces | Constraints | Non-oracle ↑ | Observed shift ↑ | Removal ↑ | Missing refs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1879 | 100.0000% | 0.00 | 2799 | 243682 | 61.5385% | 87.5000% | -28.9207% | 2 |
| cost-10x20-f50 | 506 | 25.0959% | 64.84 | 765 | 8238 | 30.7692% | 83.3333% | -201.5373% | 2 |
| cost-10x20-f80 | 512 | 25.4765% | 63.18 | 775 | 8535 | 30.7692% | 79.1667% | -142.2651% | 2 |
| cost-20x40-f50 | 497 | 24.8128% | 115.01 | 754 | 8018 | 50.0000% | 75.0000% | -237.6000% | 2 |
| cost-20x40-f80 | 501 | 25.1278% | 115.59 | 761 | 8227 | 53.8462% | 79.1667% | -158.1425% | 2 |
| length-10x20-f50 | 466 | 33.5039% | 64.12 | 939 | 14269 | 30.7692% | 83.3333% | -79.9369% | 2 |
| length-10x20-f80 | 470 | 33.6986% | 63.79 | 944 | 14433 | 26.9231% | 79.1667% | -79.8729% | 2 |
| length-20x40-f50 | 458 | 33.3366% | 116.83 | 930 | 14085 | 30.7692% | 83.3333% | -84.6245% | 2 |
| length-20x40-f80 | 461 | 33.5530% | 116.23 | 934 | 14230 | 30.7692% | 83.3333% | -84.0880% | 2 |
| presence-10x20-f50 | 506 | 25.6943% | 73.10 | 781 | 8627 | 30.7692% | 75.0000% | -123.9035% | 2 |
| presence-10x20-f80 | 512 | 26.0842% | 72.79 | 791 | 8926 | 30.7692% | 75.0000% | -231.9450% | 2 |
| presence-20x40-f50 | 497 | 25.4492% | 126.59 | 770 | 8403 | 30.7692% | 75.0000% | -135.1805% | 2 |
| presence-20x40-f80 | 501 | 25.7642% | 129.29 | 777 | 8619 | 30.7692% | 75.0000% | -136.0723% | 2 |

## Verdict

No improvement: keep disabled. Best filtered accuracy-first setting: `cost-20x40-f80`. This tests postprocessing the completed candidate pool, not online replacement/reseeding. Deleting overlapping evidence changes BP connectivity and weights and may lose unique tails; these mechanisms were not individually isolated. No2048 test or iteration increase.

Oracle reference accuracy (separate from non-oracle readout): control: 80.7692%, cost-10x20-f50: 84.6154%, cost-10x20-f80: 84.6154%, cost-20x40-f50: 84.6154%, cost-20x40-f80: 84.6154%, length-10x20-f50: 84.6154%, length-10x20-f80: 84.6154%, length-20x40-f50: 84.6154%, length-20x40-f80: 84.6154%, presence-10x20-f50: 84.6154%, presence-10x20-f80: 84.6154%, presence-20x40-f50: 84.6154%, presence-20x40-f80: 84.6154%.

Validation: native synthetic overlap tests, full native regression suites and Python config tests. The final help text, early invalid-argument checks and extra manifest-provenance field were added after the measured binary was built; no selection arithmetic changed. Per-run binary/library hashes and source patches record the exact benchmark version. No new model inference was run.

Same selected median±64/min256 BP parameters,500messages,128threads,RelWithDebInfo; one trial per setting, cached inference only. Baseline BP rerun on the identical frozen input. Retained length is total geometry, not reference coverage. Removing whole pieces may lose unique tails. Observed shift excludes missing references; non-oracle denominator26. Removal is the separate reference-conditioned score. Artifact replay not measured: do not substitute direct reference-seeded replay, which does not test this filter.

Opt-in native mode: `vc_fiber_trace_chunk deduplicate SOURCE --output TARGET --normal-manifest NORMAL --overlap-rank RANK --overlap-normal N --overlap-tangential T --overlap-fraction F --threads 128` (RANK is length, cost or presence; presence additionally requires `--fiber-manifest`). Each output records original indices, suppressor IDs, measured overlap and full source processing settings.

Approved runner: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, control action `iso20_post_overlap` (`report_only:true` to regenerate report). Defaults unchanged.

All commands/provenance/artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261003_iso20_post_overlap_v1`.
