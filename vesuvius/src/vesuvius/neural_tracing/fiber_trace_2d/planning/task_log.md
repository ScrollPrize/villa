# Short training comparison log

- Reporting refinement: re-audited classic100directions; verified same five
  scrolls, counts and reference lengths against all six modern evaluations.
  New report uses only equal-scroll arithmetic mean (20% each) for overall score;
  preserves per-scroll distributions and highlights best/tied values per metric.
  Scores: Classic5.722681, NMLraw2.482426/EMA3.053523,
  JSONraw4.930175/EMA6.307615, combinedraw5.367841/EMA6.202080mm.
  JSONEMA now ranks first. Saved docs/training_sample_comparison_equal_scroll.md;
  old pooled score omitted there. No GPU work or active-training changes.
  Independent review checked all210 per-scroll cells/highlights and seven macro
  scores against source data: no errors. git diff --check passed.

- Corrected combined601600 evaluation complete: raw3.025668mm/2571restarts;
  EMA0.99 3.986955mm/1927restarts. Both chosen at4600/588800 by test loss;
  both completed100directions and independent raw-artifact audits. All matched-
  budget per-scroll/distribution/provenance tables saved in docs/training_sample_comparison.md.
- CombinedEMA beats JSONEMA pooledmean22.0%, but gain mainlyParis4; JSONEMA
  remains better on0139/0332/1451. Do not imply uniform improvement acrossscrolls.
- Follow-up runner exited0 and launched managed continuation root1975660 from
  saved4700 rawcurrent, optimizer/EMA retained, LR.01/2Mupdates. Launch164113.
- Independent code review found no current blocker. Reusable-controller limits:
  missing continuation rawbest fails loudly even with valid parent; resume trusts
  identity-bound verified stop record. Neither applies to this completed run.

- Initial six evaluations completed: mean segment mm raw/EMA: NML1.81976/2.20086,
  JSON2.75070/3.26792, combined500k2.19369/3.69527. Combined won and resumed
  from raw3900 state (optimizer/all EMA retained), not EMA export weights.
- User corrected combined500k typo to600k. Earlier conversational promise had
  not armed a watcher; corrected now before cutoff. Identity-bound watcher
  verified601600/update4700 snapshots and stopped continuation root1798618.
- Follow-up controller reuses managed evaluation, includes parent/continuation
  best candidates and rejects beyond-cutoff snapshots; resumes raw current only
  after both audits. Independent review completed. CPU tests8passed1.60s.
- Full per-scroll/raw/EMA report requested, including pooled distributions, not
  averages of per-scroll percentiles. Two additional evaluations now running.

- Active S1a EMA run20260912_111221 at460800samples/update3600. Next scheduled evaluation above600k is4700updates=601600samples.
- Third condition repeats five-scroll; requested clarification. No speculative dataset selected.
- Reuse50fiber/100direction workflow documented in docs/benchmarks/collection_182306_latest_best_full_20260912.md.
- Independent review: timing logs precede snapshot writes. Monitor now waits
  for readable complete current raw/EMA snapshots plus consistent best metrics;
  TERM only for verified-alive recorded identity. Unknown identity fails closed.
- CPU validation: venv python -m pytest -q
  vesuvius/tests/neural_tracing/test_experiment_sample_stop.py:4passed1.42s.
- Approved run_fiber3d_experiment production-watch is active, bound to launch
  s1a_nml_128_sd1_16x8_ema-20260912_111218/root1359679 including start/boot times.
  Target600000, expected checkpoint601600. Last observed512000samples.
  Tool session55730. Result will be saved to launch/sample_stop_result.json.
- Subsequent training conditions and six evaluations are not yet executed;
  third condition clarification remains pending. Winning EMA inference export
  does not itself contain optimizer state: resume winning run's raw checkpoint.
- Clarified combinedJSON+NML500k, Paris4 JSON/NML10%each. Implemented explicit
  sampling groups/weights; ungrouped RNG sequence preserved. Independent review
  approved concrete config probabilities/heldoutsets. CPU suite29passed2skipped
  (CUDA target tests unchanged; sampler changes CPU-only), plus strictmissing
  weight case added. No GPU tests outsidemanager.
- S1a stopped successfully601600samples/update4700; result JSON verified.
  Rawbest selectedupdate2800/358400samples,testloss2.3887399733,SHA
  5b359ed861fcccb491d3cf35fd159795aabce762f46ac54237bfdd1dfd6c00c9.
  Full50fiber100direction preflightpassed; exclusive224CPU/8GPU task
  79c3e3bb-01bf-45c9-bb79-202f382e922e acquired. Evaluation in progress.
- RawS1a evaluation finished exit0, complete100directions:4341failures,
  mean1.8197608781mm,median0.0980952981mm (complete reference partitions).
- Campaign running through approved `run_fiber3d_experiment comparison`, tool
  session20448. Durable report:
  /ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/comparison.json.
  S1aEMA selecteddecay.99 at3000updates/384000samples,testloss2.4918093085;
  SHA4f7b875e116332202a5eb762c2a7229280a8b7631326e3340f898a76c523cdf7.
- Independent campaign review verified six-way audit/raw-state resume and found
  abnormal-exit cleanup gap; fixed ownership-bound stop and added CPU regression.
  Final sampler/controller suite30passed2CUDA-skipped; added campaign cleanup
  test brings controller suite to5passed1.08s. CUDA skips concern unchanged target
  kernels; actual evaluation is manager-exclusive8GPU.
- Commits9388f982a(group/configs),bc183cfc3(controller/docs). No outcomes for
  pendingJSON/combined trainings or other evaluations claimed yet.
- S1aEMA complete100directions/exit0:3572failures,mean2.2008600380mm,
  median0.0762288451mm. Both raw/EMA artifacts independently audited.
- FreshJSON-only run started:
  /ephemeral/hendrik/fiber3d_train2/runs/compare_5scroll_128_sd1_16x8_ema_600k_20260912_123713.
  Root1426267, managedcalibrationtask8dc1834e-81ac-42fe-bf3b-78637fda2a39.
  Initialtest and first9updates passed; target4700updates. Campaign owns bounded
  shutdown and will proceed through both JSON evaluations, combined training,
  both combined evaluations, then raw-state continuation of trace-metric winner.
- JSON calibration completed, adaptive task7154b406-9b78-4339-8449-bdee2cd86272
  running. First scheduled12800sample/update100 test completed37.512s.
  Campaign remains active; JSON/combined results and final continuation are
  pending, not claimed complete. No further manual start commands required.
