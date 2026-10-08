# Indexed coverage stopping:1024 pilot

Same raw40500,beam8/lookahead2,full-prefix512/iso20. Seed coverage remains40/80,angle45; only accepted-trace stopping radii vary. Same selected median64/minimum256 BP costs,500messages, 128threads,RelWithDebInfo,one run per variant,cached predictions/anchors,no GPU or2048run.

| Variant | Trace pieces | Length vs control | Trace s | Commit s | Coverage stops | BP pieces | Constraints | Non-oracle ↑ | Observed shift ↑ | Removal ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1879 | 100.00% | 56.93 | not recorded | 0 | 2799 | 243682 | 61.5385% | 87.5000% | -28.9207% |
| stop40x80 | 873 | 25.96% | 41.39 | 29.24 | 1140 | 1138 | 13691 | 26.9231% | 79.1667% | -86.9844% |
| stop20x40 | 582 | 33.24% | 42.30 | 28.26 | 754 | 1018 | 15401 | 26.9231% | 79.1667% | -92.6811% |
| stop10x20 | 593 | 33.39% | 47.44 | 31.18 | 1944 | 1038 | 15890 | 26.9231% | 79.1667% | -101.9060% |
| stop5x10 | 602 | 33.70% | 46.02 | 31.23 | 2237 | 1053 | 16381 | 38.4615% | 87.5000% | -99.2822% |
| stop2x4 | 647 | 34.37% | 47.13 | 32.75 | 2220 | 1095 | 17539 | 30.7692% | 75.0000% | -91.0150% |

## Verdict and implementation

Keep disabled by default. All tested stops reduce geometry and wall time but lose non-oracle accuracy versus control. The best stopping accuracy is5/10; no2048trial was run. Smaller graphs alone are not a quality improvement. Stopping does not create a continuation/equality link to the contacted trace in BP; whether this or changed coverage/weighting explains the regression has not been isolated.

Opt in with `--stop-at-covered-traces --covered-trace-normal N --covered-trace-tangential T`. Spatial buckets hold accepted segments; direction and sphere rejection precede normal sampling. Workers use submission watermarks and ordered commit rechecks raw prefixes before terminal reverse verification. Both sides are checked before publishing either. Commit seconds include ordinary seed suppression and indexing, not just reverse rechecks. One run per setting: these are end-to-end timings, not latency distributions.

Build/tests: approved runner with control action `build`; experiment action `iso20_coverage_stop`, optional `report_only:true`. Command: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`. Eight Python config/helper checks and native regression suites passed. No-flag OBJ geometry exactly matches historical1024. Final code also matches worker per-edge length arithmetic on arbitrary scales; the benchmark uses scale2 where this is unchanged.

Lengths are summed retained geometry, not reference coverage. Observed shift excludes missing references; non-oracle accuracy denominator26. Removal is separate reference-conditioned oracle score. Direct reference-seeded replay does not exercise accepted-trace coverage and has not been rerun/misrepresented as a new score. Artifact replay needs native collection export; not measured in this pilot. Full metrics/errors and counters in JSON.

Artifacts/configs/provenance:`/ephemeral/hendrik/fiber_crop_bench/20261003_iso20_coverage_stop_v1`. Approved runner action `iso20_coverage_stop`. Defaults unchanged.
