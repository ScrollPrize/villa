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
