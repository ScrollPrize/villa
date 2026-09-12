# Single-backward task log

- 2026-09-12: Confirmed current managed path uses three component VJPs; current
  run last20 logged window92.3global samples/s on8GPUs,median fw/bw1217ms,
  negligible loader wait. Not a matched performance experiment.
- Plan independently reviewed by review_single_backward. Counts must preserve
  nonempty sample-group semantics. Stage owned host copies before releasing
  queue slots to avoid corruption/deadlock; reuse target voxelizer on same
  device with geometry-only image view. No model forward, image H2D or aug RNG
  needed for counts. One local macro batch in host staging plus CUDA lookahead.
- User approved rejecting embedding and ordinary multi-branch losses temporarily;
  explicit TODO errors added, inference remains compatible. Conditioned mode now
  uses target-derived counts and single backward too. Objective redesign deferred.
- Initial CPU fixture incorrectly combined CP-only data with arc-length mode;
  corrected fixture. Approved regression run20260912_080545_964494:
  28passed/15CUDA skips in28.42s. Added zero-work distributed test subsequently.
- CUDA suite20260912_080651_075055 was cancelled while waiting for manager
  release; no GPU tests ran. Test cap20GiB/GPU; short isolated speed comparison uses
  1GPU, B4,128cube,GN/stride4/BF16,5warmup+10measured iterations per path.
- Expanded approved CPU suite20260912_083825_640311: 37 passed,15 CUDA skips,
  31.88s. Includes early unsupported-mode errors and conditioned target counts.
- Host process view confirms training root1200528 exists (sandbox process view
  incorrectly suggested it was gone). Exact live GDB stack not yet provided.
  Inspection found cleanup can acquire dead-owner counter locks after worker
  termination, has no KILL fallback and joins queue feeders without a deadline.
  Fixing these plus independent stop escalation; not resetting manager ownership.
- Independent review required persistent session ownership (not periodic PPID
  discovery), atomic registration and a separate signal path. Extracted cohort
  session inspection for reuse; managed broker workers own sessions and register
  at spawn. Guard readiness/health checks and dead-session pruning added.
- Orphan test exposed scan/prune ordering race: now inspect leader liveness before
  scanning sessions. Approved run20260912_090743_399930: stop-guard3passed15.75s.
  Prior run had59passes/2skips and that1failure; corrected rather than waived.
- Full CUDA measurement still blocked by live old root1200528 holding allocation;
  user requested GDB capture asynchronously. No release/reset or unauthorized GPU use.
- User stack file contained ptrace attachment failure, not stacks. Host check
  confirmed GPU workers gone/GPUs empty; user approved KILL of root1200528 only.
  Shared manager untouched; dead-task reconciliation unblocked exclusive tests.
  Exact original blocked native site cannot be proven from that failed capture.
- Run20260912_090846_805192 shutdown61passed/2skips118.43s. Existing loss tests
  initially had3 expected feature-availability failures; changed multibranch case
  to assert error and retained mask/negative normalization coverage with one branch.
- Approved run20260912_095321_965757: loss regression13passed/1skip2.83s;
  single-backward38passed/15GPUskips31.94s; manager-exclusive cuda-arc3passed11.78s.
  Matched one-H100 model/loss/backward test: S1a config UNet, synthetic array/trace
  fixture with random volume, B4,128cube,stride4,compiled installed PyTorch/native
  stack,BF16;5warm+10measured each. Component mean246.2056ms,p50246.2810,
  min245.4356,max246.6364,16.2466samples/s. Single mean102.7572ms,p50102.7595,
  min102.6342,max102.8985,38.9267samples/s (2.396x). BF16 gradient relativeL2
  .0053636;FP32 CPU full-batch/Adam equivalence passes. No end-to-end speed claim:
  benchmark excludes loading/target-prepass/optimizer-step and distributed reduction.
- Same approved invocation: manager-exclusive two-GPU training/evaluation handoff
  and changed-rank-count resume2passed43.96s. Driver peaks: model-only test
  12,826,181,632bytes on GPU0; cohort1,314,914,304/1,367,343,104bytes on GPU0/1,
  within authorized20GiB per GPU. All test leases released (launcher exit0).
- Two old multi-branch routing loss oracles are retained as strict expected
  ValueErrors pending the explicitly deferred normalization fix. Independent
  direct startup/evaluation error tests cover the new unsupported behavior.
  Final expanded loss suite20260912_095643_242639:13passed,1GPUskip,2expected
  unsupported-mode failures in3.18s. `git diff --check` passes.
- Reproduction command for all recorded validations:
  `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests`.
  Each artifact directory's control.json/results.json records selected suites,
  exact pytest argv, timing and GPU peaks. No installation/build change needed.
