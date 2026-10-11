# Joint full passes and reference metrics (archived)

- Implemented uv_full_pass_plan.md after independent agent review.
- Joint solver defaults to full_pass traversal, passes=10. Each pass visits
  every event and every node once through balanced seeded permutations. Internal
  Adam updates still batch for GPU memory; all geometry and probabilities train.
- All continuation links are evaluated every update; no random link sampling.
  Tail normalization preserves relative event/node objective weights.
- active_fraction must be zero. Historical sampled behavior is available only
  explicitly with traversal=sampled; old active-sampling configs fail clearly.
- Optional reference_projection points to existing projection.npz. Identity
  checked against source_id/start/end. Metrics recorded at every full pass:
  source-fiber L1, inclusive ±2 percentage, continuous median global offset,
  retained/matched counts. Actual fitted curve positions are used, not bucket IDs.
- Reference comparisons are approximate cached surface associations, not truth;
  rejected/unmatched fibers excluded from accuracy but counted explicitly.
- Production launch: run_uv_fullpass151_{ccw,cw}_joint.json via approved
  run_classic_fiber_evaluation.py crop-benchmark runner. Ten weak full passes,
  User cancelled the subsequent strong phase: ONLY these ten current joint
  passes per direction are authorized. Do not launch the _sign.json configs.
  Both request shared (exclusive=false) 24 GiB VRAM on separate GPUs.
- Validation: 31 focused tests passed; git diff --check clean. Implementation
  reviewed independently. Added AFV hash/signature validation to projection loading.
- Full-pass loss is the mean online objective across its updates, not the last
  batch and not a frozen-state global objective. Reference scores are at pass end.
- Follow-up: add equal-fiber fit/reference q10/q90 ranges after the same global
  offset. Running processes already imported old metric code: compute these
  from final saved states; intermediate ranges cannot be recovered (no per-pass
  checkpoints). No optimizer restart. Added metrics-only CPU report path.
- Reference orientation checked via approved runner report_reference_direction.json:
  increasing surface column is CCW in atan2(y-cy,x-cx), 99.97443% tangential weight
  over182808 valid sampled segments. Raster-view orientation can be inverted.
- Current run inherited rejection_warmup=0.5; gradients to broken logits were
  held for first5passes. This was disclosed; user asks to finish current run only.
- Completed CCW/CW10passes in1499.2/1508.9seconds. FinalL1=11.5170/11.5797,
  within±2=13.1949%/12.9902%, retainedlength86.3351%/86.3099%.
  Full-graph sign violations5.0359%/5.0365%. Final fit q10/q90=48/121 after+11
  offset, reference42/124 on same matched fibers. Metrics-only approved runner
  reproduces final L1 exactly. Full tables saved uv_fullpass151_results.md.
- Quantile regression tests:15passed. No strong run launched; both GPU jobs exit0.
