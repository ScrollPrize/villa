# Implementation plan

1. Extract shared sample queue construction from training iterator and broker.
   Train/test source selection and canonical RNG indices must be explicit and
   distinct from transport slot indices. Reuse bounded shared slots, runnable-work
   scheduling, single-core workers and CUDA preparation.
2. Route dense evaluation through this queue, including managed wave/resume paths.
   Admit work in execution order; training lookahead must never occupy all credits
   needed for earlier evaluation. Reuse workers. No synchronous production fallback.
3. Start exclusively using existing manager protocol. Measure real initial test and
   real optimizer updates while ascending through conservative physical microbatch
   sizes bounded by logical batch. Preserve state/RNG, never discard updates.
   Include test and asynchronous preparation peaks; aggregate ranks, add reserve.
   Only measured sizes/devices may inform subsequent adaptive grants.
4. Persist the profile and transition at a safe joined/released boundary.
   User approved two registrations within one training run, with RAM state
   retained, preserving protocol v1's immutable registration capabilities.
   Do not mutate protocol fields or silently change the proposed workflow.
   Log waiting, observed peaks, and registration/request transition.
5. CPU tests: deterministic source/index mapping, tiny bounded queue transitions,
   retained leases, partial-test resume, startup exclusivity, profile validation,
   request transition and optimizer/sample-state preservation.
6. GPU checks via approved reusable benchmark launcher, all GPUs exclusively held,
   bounded runtime. Compare fixed-sample metrics and cold/warm throughput; exercise
   real updates and adaptive transition. Do not replace production by benchmark code.

## Independent review corrections
- Preserve a separately measured evaluation microbatch cap unless all subsequently
  used evaluation sizes have actually been measured during real evaluation.
- Include initialized optimizer/banks/asynchronous preparation in measured peaks.
- OOM fails loudly; do not retry/discard a partially executed real optimizer update.
- Explicit admission boundaries prevent future training consuming test credits.
- Replace conflicting old specs for separate test pools and manual calibration.

## Spec update
Replace optional fast-loader/legacy eval fallback with shared queued train/test.
Document mixed-source ordering and real-update automatic memory calibration.
Preserve logical batch loss normalization and BF16.

## Docs updates
Update cooperative training single-command workflow, profile storage/limits and
reproducible performance results. Remove requirement for standalone calibration.

## Changelog
Record unified train/test loading and automatic real-update startup calibration.
