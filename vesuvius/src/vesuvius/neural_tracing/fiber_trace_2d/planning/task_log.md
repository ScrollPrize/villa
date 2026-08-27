# Task Log: Explicit Mixed-state fiber belief propagation

## 2026-08-27

- Started from committed binary sum-product BP at `54b24dbb8`.
- Chose a separate ternary inference mode so the binary min-sum and
  sum-product experiments remain exactly reproducible.
- Model Mixed as the established defect/Broken meaning: it disables the H/V
  orientation factor and pays a configurable penalty per incident merged link.
  It is not a third physical direction.
- Independent review required the Mixed penalty to scale by the raw
  measurement count represented by a merged factor, explicit normalized
  message gauges, direct `P(Mixed)` diagnostics, separate tie handling, and
  tests for unseeded gauge symmetry and duplicate measurements. The plan and
  implementation were corrected before validation.
- Implemented normalized three-entry log messages, exact hard-H seeding,
  normalized V/Mixed/H node marginals, explicit orientation projection, direct
  Mixed AUROC/confusion summaries, probability summaries, and separate Mixed
  probability OBJ bands. Existing binary inference output remains unchanged.
- Focused build and validation:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiberlet_crop_trace -j32`,
  `volume-cartographer/build/bin/test_fiberlet_crop_trace`, and
  `git diff --check` passed. The suite now has 64 passing cases.
- Centered-384 evaluation used 179 fibers (50 Direction1, 45 Direction2, 84
  Mixed), 1324 perpendicular factors, `T=2.5`, and the full admitted Mixed
  cohort. Mixed-cost sweep results for direct `P(Mixed)` AUROC were:
  `0.0: 0.398434`, `0.1: 0.416855`, `0.2: 0.695802`, `0.25: 0.744173`,
  `0.275: 0.742419`, `0.3: 0.747807`, `0.325: 0.754073`,
  `0.35: 0.746429`, `0.375: 0.730514`, `0.4: 0.720363`,
  `0.5: 0.693672`, `0.75: 0.662719`, and `1.0: 0.647431`.
- The best tested cost `0.325` improves Mixed ranking over the committed best
  binary consistency proxy (`0.694641` AUROC). Mean/median `P(Mixed)` were
  `0.223164/0.236099` for Direction1, `0.226372/0.234011` for Direction2, and
  `0.299017/0.333333` for Mixed. Argmax is conservative: 9/84 Mixed fibers
  select Mixed, no trusted fiber selects Mixed, and four exact ties include
  one unsupported trusted fiber plus three unsupported Mixed fibers.
- Kept the experimental default Mixed cost at `0.5`: selecting `0.325` from a
  single crop would overfit the diagnostic. The measured command can pass
  `--bp-mixed-cost 0.325` explicitly.
- Follow-up evaluation on the existing 1024-cube, 500-trace artifact used
  `T=2.5`, Mixed cost `0.325`, 4941 perpendicular factors, and the noisier
  0.9-dominance direction reference (209 Direction1, 157 Direction2, 134
  Mixed). It converged in 164 iterations and 0.173 solver seconds.
  `P(Mixed)` AUROC was `0.803727`. Ternary argmax assigned 26/134 Mixed
  references and only 1/366 trusted references to Mixed. With the seed making
  Direction1 H, trusted argmax contained two H/V reversals, both among
  Direction2; Direction1 had 205 H, one Mixed, and three ties, while Direction2
  had 154 V, two H, and one tie. The generated initial-direction, orientation-
  projection, and Mixed-probability OBJ layers are under
  `data/workdir3/fiber-crop-1024/bp-mixed/`.
- The first 1024 run optimized Mixed ranking rather than three-way argmax and
  was rejected as the final classifier because only 26/134 Mixed references
  selected Mixed. A coarse and refined deterministic sweep then optimized both
  mean per-class recall and minimum per-class recall. The best mean-recall
  point was `T=1.0,cost=0.14` (H 199/209, V 150/157, Mixed 72/134), while the
  selected max-min point was `T=2.4,cost=0.125` (H 157/209, V 111/157, Mixed
  94/134). The selected recalls are 75.1%, 70.7%, and 70.1%, respectively;
  `P(Mixed)` AUROC is `0.812943`, and there are no H/V reversals. The trusted
  errors are 49 H-to-Mixed, 45 V-to-Mixed, and four exact ties; Mixed errors
  are 22 Mixed-to-V and 18 Mixed-to-H.
- Final 1024 artifacts overwrite the main visualization basename under
  `data/workdir3/fiber-crop-1024/` as `fibers_initial_*`,
  `fibers_bp_sum_product_mixed_*`, and the consistency CSV. The superseded
  `bp-mixed/` subdirectory and temporary sweep outputs were removed.
