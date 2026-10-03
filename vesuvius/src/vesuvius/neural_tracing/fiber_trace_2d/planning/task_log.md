# Raw best checkpoint by reviewed-trace restarts

- Existing `best.pt` independently minimizes raw dense `test/loss_total`; EMA99
  and EMA999 likewise minimize their matching EMA dense test losses.
- Periodic reviewed tracing evaluates the raw `current.pt` (or startup
  `best.pt`) in both managed and unmanaged paths, but currently only records
  metrics and does not retain the best trace-performing raw checkpoint.
- Planned behavior: `best_reviewed_trace.pt` strictly minimizes
  `restarts_combined`; ties retain the earlier checkpoint. Existing checkpoint
  names and selection rules remain unchanged.
- Independent review identified that mutable `current.pt` cannot safely be used
  after an interrupted trace finalization. The implementation will freeze and
  hash the exact evaluated raw checkpoint before launch, bind a stable benchmark
  identity, write completion only after cleanup, and reconcile interrupted
  successful finalization idempotently.
- Added shared managed/unmanaged finalization which freezes the evaluated raw
  checkpoint, validates trace completion and benchmark identity, and strictly
  selects `snapshots/best_reviewed_trace.pt` by combined restart count. Existing
  dense raw and EMA selection remain untouched; `best_trace` resolves the new
  snapshot in the experiment runner.
- Added atomic/idempotent recovery and tests for improvement/ties, malformed or
  mutated inputs, identity changes, incomplete/revoked attempts, and interrupted
  promotion. No GPU benchmark is required because selection consumes the
  already-validated reviewed-trace result.
- Validation: focused selection/runner tests passed (55 tests), cooperative
  integration passed (11 tests, 2 skipped), Python compilation passed, and
  `git diff --check` passed. All validation was CPU-only and left the live
  training process untouched.
