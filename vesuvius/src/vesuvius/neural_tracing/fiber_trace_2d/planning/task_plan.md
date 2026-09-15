# Fixed test crops

## Current correction
Stop active run. Mark test observations explicitly independent and use uniform
ellipse cap without border taper or endpoint pinning in the shared solver.
Preserve training constraints. Test scalar/batched parity, border/end movement,
test observation transport and existing managed CUDA path. Independently review.
Restart from original non-refinement update11800, not either refinement run.
Spec/docs update: remove stated test taper/end locks; retain step cap/tether.

1. Commit old state (5ba084bfd); keep training behavior unchanged.
2. Test transforms use original GT; lazily register per-sample/transform local
   curves in existing store/archive. Old whole-test updates do not seed them.
3. Share raw/EMA scalar logging. Preserve names and checkpoint selection semantics.
4. Test crop stability, target changes, overlap isolation, resume and metric parity;
   approved CPU runner and manager-owned CUDA smoke. Independent review.
5. Stop incorrectly resumed refinement run; resume original non-refinement run
   5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436/current raw at11800.
   Verify no refinement archive/config in source checkpoint; keep optimizer/EMA,
   initialize fresh refinement geometry. Check manager launch and resume log.

## Spec update
Test geometry is independent per fixed GT crop; same raw/EMA targets per visit.
Existing nearest connected section solver scope and training cap/taper
behavior remain unchanged. Archive local test state, not merged test fibers.

## Docs updates
Update refinement docs, specs, changelog and task log with transition and launch.
