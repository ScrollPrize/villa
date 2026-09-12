# Short training comparison log

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
