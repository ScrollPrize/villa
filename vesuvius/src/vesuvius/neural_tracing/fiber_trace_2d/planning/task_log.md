# Bounded parallel failure search

Prior run stopped for scheduler correction. Baseline native~564% CPU despite64 workers. Source inspection found eager301134 guarded8MiB stacks, longest-first order and no native pruning; previous claim that cancellation was already active was incorrect. Need bound active execution and enable safe paired-audit pruning. No inference algorithm changes.

Implemented bounded8 contexts/OS worker with shared atomic job admission and
stack reuse; global trials materialize geometry only when admitted, then free
it after artifact publication. Global trials sorted ascending distance; ordinary
evaluation keeps its original ordering. Paired valid n/n+100 audits set per
candidate/direction cutoff; shorter trials retained. Already-running trials
finish (no unsafe forced stack interruption); longer queued trials skipped.
This is an explicit limitation vs immediate in-flight cancellation.

Native audit duplicates the existing Python first-failure/grace predicate for
online pruning; every emitted native decision is checked against Python's saved
geometry audit before acceptance. Mismatch fails closed. Completed/cancelled
counts and unique artifact counts validated; skipped trials are not negatives.
No generic benchmark restart statistics are fabricated for cancelled trials.

Removed extraction's loader/accumulator/flush/IO/slot overrides. Producer
inference settings preserved verbatim. Manager/tracing CPU budget explicitly192,
all8 GPUs unchanged. Native build via approved runner action build; workspace
RelWithDebInfo. Dependency script named in root instructions is absent; existing
native build runner used, no installs.

Tests: bounded301134 trivial jobs~0.038-0.040s (8 OS workers, g++13 -O2), not a
tracing speed claim. Native23 replay+86 tracer+18 reference tests pass. Python47
extractor/namespace tests pass with villa3 source path. Independent agent review
requested accounting checks, added. git diff --check clean.

Resumed via approved runner PID101496 after verified old-session76919 orphan
cleanup found0 survivors. Existing inference and prior attempts retained;
implementation/native hashes revisioned by existing resume logic.

Live validation13:30:20 UTC: native trial progress17200/301134,360 pruned;
inference5535 tiles/1169 published chunks in109.4s. Native lifetimeCPU4044%
at55s (baseline564% at313s; observational, not controlled speedup comparison).
All8 GPUs active, sampled65–100% utilization. Hundreds of contexts waiting for
inference explain part of remaining CPU headroom; normal-cache remote fetches
also visible. Sampled500 real emitted traces: native/Python audit0 mismatches,
12 reproduced. Extraction remains running; complete-stage runtime not yet known.

User requested stop/resume on GPUs0-5. Stopped session101496 with approved
runner; orphan check found0 survivors. Updated control and both reproduction
and highres-fill device settings only; CPU192 and inference tuning unchanged.
Restart PID125793. Important limitation reported: existing resume imports trial
audits only after a full native batch, so interrupted partial-batch artifacts
are retained but not automatically reused; that batch replays. Published
inference and completed evaluation inputs are reused. No scheduler changes in
this GPU-selection turn.
