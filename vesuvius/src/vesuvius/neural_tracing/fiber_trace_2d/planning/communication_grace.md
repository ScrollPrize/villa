# Manager communication grace

## Task and implementation
User requests logged communication errors, continued normal training for a
60-second outage, then GPU-free pause with periodic reconnection and automatic
resume. Exclusive startup calibration may finish despite communication loss.

Add opt-in supervisor recovery policy: Fiber adaptive clients use60s grace and5s
retry interval; exclusive non-pausable calibration uses no outage deadline.
Only already-authorized cohorts can advance offline. Never create/authorize new
workers offline. Retain reservation, generation and task token; reconnect the
same task. Known release directives, epoch changes, explicit stop, fatal errors
and an already-started drain override grace. Never reset the outage clock on a
successful hello alone. At expiry stop admission, drain/join, retain RAM state,
then reconcile release/regrant before resume. Log actual exception and recovery.

## Spec update
This user-authorized recovery amendment allows bounded offline work for pausable
clients (previous rule required immediate safe-boundary release). No wire/schema
fields or allocation rules change; EOF never frees manager reservations.

## Docs updates
Document policy in protocol recovery and Fiber cooperative training docs; record
task results/changelog. Generic supervisor retains immediate-drain default.

## Tests
Fake clock and CPU cohorts: short outage preserves workers and advances units;
repeated failures do not reset deadline; expiry drains and resumes state; retry
cadence; lost initial applied reply never authorizes offline; stop/release/epoch
override grace; exclusive calibration continues beyond60s. Existing manager tests.
No GPU work or running-job changes.
