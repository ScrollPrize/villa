# Graceful cooperative maintenance under allocation changes

- Production run `...sideband1...20261002_075011` failed after its update-42600
  dense test when manager reallocation began in the same supervisor tick as a
  reviewed-trace maintenance boundary.
- Immediate exception: `RuntimeError: cohort is not at a maintenance boundary`.
  The model/data path was healthy and update-42600 raw/EMA checkpoints exist.
- Root design defect: maintenance was treated as immediately executable and
  cohort-local, while manager allocation changes are asynchronous and valid at
  every boundary. Active trace-child ownership also lacked an explicit
  revocation/retry transition.
- Independent review required a supervisor-owned synchronous mailbox rather
  than post-tick polling, exact epoch/generation/grant attempt ownership, exact
  GPU UUID binding, verified process-group cleanup on revocation, staged output
  promotion, fatal genuine child failures, and forced interleaving validation.
- Implemented the mailbox and exact request deduplication. Every cohort shares
  it, replacement cohorts are held while it is pending, and an allocation-
  draining cohort may close without consuming the request.
- Trace attempts are tied to exact epoch/generation/grant tokens and exact GPU
  UUID visibility. Active-attempt state blocks release acknowledgement until
  the owned process group is verified reaped. Revocation cleans the transient
  prediction namespace, preserves the incomplete staged attempt, and retries;
  genuine child failure is fatal. Completed attempts promote atomically.
- CPU validation: cooperative/reviewed/startup plus manager supervisor/policy
  suites: `92 passed, 2 skipped` in 43.73 s. Python compilation and
  `git diff --check` passed.
- Production validation resumed the failed checkpoint at update 42600. At the
  update-42700 reviewed-trace boundary, attempt 1 started under generation 1;
  a forced manager pause caused it to be process-group reaped before release,
  leaving the request pending. After resume/regrant as generation 3, attempt 2
  ran the same step request and atomically promoted `step_00042700` with 21
  combined restarts and 25.944478 mm implied length. Training then advanced
  through updates 42701-42704, proving the pending boundary was neither lost
  nor duplicated.
