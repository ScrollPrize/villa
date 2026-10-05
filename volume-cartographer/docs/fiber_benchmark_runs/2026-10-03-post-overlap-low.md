# Post-fw/bw piece overlap suppression —1024 pilot

Frozen1879 raw40500 beam8/full-prefix512/iso20 pieces. No early stopping or retracing. Every retained reverse-filter split is independent. Global quality-ranked whole-piece suppression; ties use length then source ordinal. Suppressed pieces cannot suppress others.

Collision:0%,10%,20% of the SHORTER piece;0% requires at least one positive sampled match, never disjoint pairs. Ellipses10/20,20/40 first; then isotropic10/10,20/20. Arc-weighted midpoint samples at most4base voxels apart, shared normal ellipse, unoriented angle45°. Presence is length-weighted shared prediction sampling (max valid option), not seed presence. Cost is saved tracer loss/path length, lower wins. No GT is used in selection.

| Variant | Pieces | Retained length | Postprocess s | BP pieces | Constraints | Non-oracle ↑ | Observed shift ↑ | Removal ↑ | Missing refs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1879 | 100.0000% | 0.00 | 2799 | 243682 | 61.5385% | 87.5000% | -28.9207% | 2 |
| cost-10x10-f0 | 506 | 24.9548% | 0.17 | 764 | 8174 | 30.7692% | 83.3333% | -197.3445% | 2 |
| cost-10x10-f10 | 507 | 25.0177% | 0.17 | 765 | 8205 | 30.7692% | 83.3333% | -198.4722% | 2 |
| cost-10x10-f20 | 509 | 25.1225% | 0.17 | 767 | 8276 | 30.7692% | 83.3333% | -200.5084% | 2 |
| cost-10x20-f0 | 501 | 24.8357% | 0.17 | 759 | 8063 | 30.7692% | 83.3333% | -193.9482% | 2 |
| cost-10x20-f10 | 502 | 24.9033% | 0.17 | 760 | 8125 | 30.7692% | 83.3333% | -204.9925% | 2 |
| cost-10x20-f20 | 506 | 25.0765% | 0.17 | 764 | 8235 | 30.7692% | 83.3333% | -201.4275% | 2 |
| cost-20x20-f0 | 493 | 24.4176% | 0.17 | 747 | 7765 | 34.6154% | 75.0000% | -193.0189% | 2 |
| cost-20x20-f10 | 499 | 24.8100% | 0.17 | 756 | 8040 | 34.6154% | 66.6667% | -232.5062% | 2 |
| cost-20x20-f20 | 502 | 25.0137% | 0.18 | 760 | 8174 | 30.7692% | 83.3333% | -207.7560% | 2 |
| cost-20x40-f0 | 477 | 23.2796% | 0.17 | 715 | 7022 | 34.6154% | 50.0000% | -187.7869% | 2 |
| cost-20x40-f10 | 483 | 23.7805% | 0.17 | 726 | 7384 | 38.4615% | 50.0000% | -229.7901% | 2 |
| cost-20x40-f20 | 490 | 24.3836% | 0.17 | 741 | 7715 | 34.6154% | 58.3333% | -245.6541% | 2 |
| length-10x10-f0 | 468 | 33.3608% | 35.85 | 941 | 14168 | 26.9231% | 79.1667% | -79.3645% | 2 |
| length-10x10-f10 | 468 | 33.3622% | 0.17 | 941 | 14169 | 26.9231% | 79.1667% | -79.3771% | 2 |
| length-10x10-f20 | 469 | 33.4080% | 0.17 | 942 | 14213 | 26.9231% | 79.1667% | -80.0938% | 2 |
| length-10x20-f0 | 462 | 33.2715% | 0.18 | 935 | 14038 | 30.7692% | 83.3333% | -78.4643% | 2 |
| length-10x20-f10 | 463 | 33.3413% | 0.17 | 936 | 14110 | 30.7692% | 83.3333% | -79.3796% | 2 |
| length-10x20-f20 | 465 | 33.3927% | 0.17 | 938 | 14163 | 30.7692% | 83.3333% | -80.0992% | 2 |
| length-20x20-f0 | 456 | 32.9157% | 62.06 | 925 | 13728 | 30.7692% | 83.3333% | -76.6113% | 2 |
| length-20x20-f10 | 460 | 33.2840% | 0.17 | 933 | 14043 | 30.7692% | 83.3333% | -78.9145% | 2 |
| length-20x20-f20 | 462 | 33.3526% | 0.17 | 935 | 14107 | 30.7692% | 83.3333% | -79.8903% | 2 |
| length-20x40-f0 | 435 | 31.5782% | 112.68 | 886 | 12636 | 50.0000% | 79.1667% | -82.9448% | 2 |
| length-20x40-f10 | 445 | 32.4070% | 0.17 | 905 | 13335 | 30.7692% | 83.3333% | -88.8810% | 2 |
| length-20x40-f20 | 451 | 32.6998% | 0.17 | 916 | 13565 | 30.7692% | 83.3333% | -91.3258% | 2 |
| presence-10x10-f0 | 506 | 25.5314% | 7.04 | 779 | 8526 | 30.7692% | 75.0000% | -204.2827% | 2 |
| presence-10x10-f10 | 507 | 25.6258% | 7.15 | 781 | 8592 | 30.7692% | 75.0000% | -203.9264% | 2 |
| presence-10x10-f20 | 509 | 25.7298% | 6.92 | 783 | 8663 | 30.7692% | 75.0000% | -204.1784% | 2 |
| presence-10x20-f0 | 501 | 25.4030% | 7.32 | 774 | 8420 | 30.7692% | 75.0000% | -202.9867% | 2 |
| presence-10x20-f10 | 502 | 25.5021% | 7.13 | 776 | 8510 | 30.7692% | 75.0000% | -203.6033% | 2 |
| presence-10x20-f20 | 506 | 25.6744% | 7.17 | 780 | 8620 | 30.7692% | 75.0000% | -204.7013% | 2 |
| presence-20x20-f0 | 493 | 25.0132% | 7.36 | 762 | 8121 | 30.7692% | 75.0000% | -226.4068% | 2 |
| presence-20x20-f10 | 499 | 25.4371% | 6.94 | 772 | 8430 | 30.7692% | 75.0000% | -203.8933% | 2 |
| presence-20x20-f20 | 502 | 25.6399% | 7.24 | 776 | 8565 | 30.7692% | 75.0000% | -204.5875% | 2 |
| presence-20x40-f0 | 475 | 23.8286% | 7.19 | 728 | 7331 | 42.3077% | 70.8333% | -148.4243% | 2 |
| presence-20x40-f10 | 482 | 24.3614% | 6.90 | 742 | 7730 | 38.4615% | 58.3333% | -219.0260% | 2 |
| presence-20x40-f20 | 490 | 25.0877% | 7.08 | 760 | 8140 | 42.3077% | 66.6667% | -120.6560% | 2 |

## Verdict

No improvement: keep disabled. Best filtered accuracy-first setting: `length-20x40-f0`. This tests postprocessing the completed candidate pool, not online replacement/reseeding. Deleting overlapping evidence changes BP connectivity and weights and may lose unique tails; these mechanisms were not individually isolated. No2048 test or iteration increase.

Oracle reference accuracy (separate from non-oracle readout): control: 80.7692%, cost-10x10-f0: 84.6154%, cost-10x10-f10: 84.6154%, cost-10x10-f20: 84.6154%, cost-10x20-f0: 84.6154%, cost-10x20-f10: 84.6154%, cost-10x20-f20: 84.6154%, cost-20x20-f0: 84.6154%, cost-20x20-f10: 84.6154%, cost-20x20-f20: 84.6154%, cost-20x40-f0: 84.6154%, cost-20x40-f10: 84.6154%, cost-20x40-f20: 84.6154%, length-10x10-f0: 84.6154%, length-10x10-f10: 84.6154%, length-10x10-f20: 84.6154%, length-10x20-f0: 84.6154%, length-10x20-f10: 84.6154%, length-10x20-f20: 84.6154%, length-20x20-f0: 84.6154%, length-20x20-f10: 84.6154%, length-20x20-f20: 84.6154%, length-20x40-f0: 88.4615%, length-20x40-f10: 88.4615%, length-20x40-f20: 88.4615%, presence-10x10-f0: 84.6154%, presence-10x10-f10: 84.6154%, presence-10x10-f20: 84.6154%, presence-10x20-f0: 84.6154%, presence-10x20-f10: 84.6154%, presence-10x20-f20: 84.6154%, presence-20x20-f0: 88.4615%, presence-20x20-f10: 84.6154%, presence-20x20-f20: 84.6154%, presence-20x40-f0: 84.6154%, presence-20x40-f10: 84.6154%, presence-20x40-f20: 84.6154%.

Validation: native synthetic overlap tests, full native regression suites and Python config tests. Per-run binary/library hashes and source patches record the exact benchmark version. No new model inference was run.

Same selected median±64/min256 BP parameters,500messages,128threads,RelWithDebInfo; one trial per setting, cached inference only. Baseline BP rerun on the identical frozen input. Retained length is total geometry, not reference coverage. Removing whole pieces may lose unique tails. Observed shift excludes missing references; non-oracle denominator26. Removal is the separate reference-conditioned score. Artifact replay not measured: do not substitute direct reference-seeded replay, which does not test this filter.

Opt-in native mode: `vc_fiber_trace_chunk deduplicate SOURCE --output TARGET --normal-manifest NORMAL --overlap-rank RANK --overlap-normal N --overlap-tangential T --overlap-fraction F --threads 128` (RANK is length, cost or presence; presence additionally requires `--fiber-manifest`). Each output records original indices, suppressor IDs, measured overlap and full source processing settings.

Approved runner: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, control action `iso20_post_overlap` with `low:true` (`report_only:true` to regenerate report). Defaults unchanged.

All commands/provenance/artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261003_iso20_post_overlap_low_v1`.

## Best ranking per corridor and threshold

Selected by non-oracle accuracy, then observed shift, then removal score. All individual trials remain in the table above; this is not a new tuned run.

| Corridor | Overlap | Keep by | Pieces | Non-oracle ↑ | Shift ↑ | Removal ↑ |
|---|---:|---|---:|---:|---:|---:|
| 10/20 | 0% | length | 462 | 30.7692% | 83.3333% | -78.4643% |
| 10/20 | 10% | length | 463 | 30.7692% | 83.3333% | -79.3796% |
| 10/20 | 20% | length | 465 | 30.7692% | 83.3333% | -80.0992% |
| 20/40 | 0% | length | 435 | 50.0000% | 79.1667% | -82.9448% |
| 20/40 | 10% | presence | 482 | 38.4615% | 58.3333% | -219.0260% |
| 20/40 | 20% | presence | 490 | 42.3077% | 66.6667% | -120.6560% |
| 10/10 | 0% | cost | 506 | 30.7692% | 83.3333% | -197.3445% |
| 10/10 | 10% | cost | 507 | 30.7692% | 83.3333% | -198.4722% |
| 10/10 | 20% | cost | 509 | 30.7692% | 83.3333% | -200.5084% |
| 20/20 | 0% | cost | 493 | 34.6154% | 75.0000% | -193.0189% |
| 20/20 | 10% | cost | 499 | 34.6154% | 66.6667% | -232.5062% |
| 20/20 | 20% | length | 462 | 30.7692% | 83.3333% | -79.8903% |

## Historical50/80% controls (saved, not rerun)

| Variant | Pieces | Non-oracle ↑ | Observed shift ↑ | Removal ↑ |
|---|---:|---:|---:|---:|
| control | 1879 | 61.5385% | 87.5000% | -28.9207% |
| cost-10x20-f50 | 506 | 30.7692% | 83.3333% | -201.5373% |
| cost-10x20-f80 | 512 | 30.7692% | 79.1667% | -142.2651% |
| cost-20x40-f50 | 497 | 50.0000% | 75.0000% | -237.6000% |
| cost-20x40-f80 | 501 | 53.8462% | 79.1667% | -158.1425% |
| length-10x20-f50 | 466 | 30.7692% | 83.3333% | -79.9369% |
| length-10x20-f80 | 470 | 26.9231% | 79.1667% | -79.8729% |
| length-20x40-f50 | 458 | 30.7692% | 83.3333% | -84.6245% |
| length-20x40-f80 | 461 | 30.7692% | 83.3333% | -84.0880% |
| presence-10x20-f50 | 506 | 30.7692% | 75.0000% | -123.9035% |
| presence-10x20-f80 | 512 | 30.7692% | 75.0000% | -231.9450% |
| presence-20x40-f50 | 497 | 30.7692% | 75.0000% | -135.1805% |
| presence-20x40-f80 | 501 | 30.7692% | 75.0000% | -136.0723% |

Overlap graph cache stores all positive pair fractions and exact ordered input geometry. Key includes normal identity/contents, scale1, radii, angle, sample spacing and contract version. Normal chunks are assumed immutable as in the existing benchmark caches. Ranking and threshold do not alter measured geometry. Each new variant still performs its own selection and BP solves. An80% length control checks exact survivor IDs against the prior experiment before the sweep. Cached-filter timings exclude the first measurement; inspect overlap_cache_hit in JSON.
