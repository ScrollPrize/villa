# Short raw/EMA training comparison

Comparison: current S1a600k, fresh five-scroll JSON600k, combined JSON+S1a NML500k.
Counts refer to samples, not optimizer updates.
Global128 and test interval100 round the first cap to601600samples/update4700.
Combined cap is499200samples/update3900. New run configs are
train_compare_5scroll_ema_600k.json and train_compare_5scroll_s1a_ema_500k.json.
Both start from scratch at LR.01, warmup100, EMA.99/.999, regular run directory.
Combined Paris4 JSON directories share sampling_group=json, NML usesgroup=nml;
explicit weight1each means10%ofall samples each. Every other scroll gets20%.
Within groups, sampling remains proportional to fiber arclength.

The approved experiment controller accepts `production-watch`. Its control JSON
`production_stop` contains an explicit `launch` directory and positive `samples`
limit. It binds the launch's PID/start/boot identity, not `latest`, and waits for
completed raw/EMA current snapshots and consistent best snapshots before TERM.
The trainer handles bounded worker shutdown. An unverifiable identity or failed
shutdown is an error, never permission to kill an unrelated process. Successful
stop writes `sample_stop_result.json` in that launch directory. This is a
checkpoint-boundary stop, not an exact mid-macrobatch sample cap. Keep the monitor
alive until completion; a live trainer does not reread edited max_steps values.

Select best raw by current test loss, and one best EMA across the configured
decays by its current test loss. Freeze files before inference and retain hashes.
Run the existing locked50fiber/100direction bidirectional evaluator on all six
checkpoints; compare complete-reference mean segment length and per-scroll
failures. Training test losses from different datasets are not directly comparable.
Resume the winning run from its resumable raw checkpoint (including EMA state),
not its inference-only EMA export. Full comparison results are still pending.

`run_fiber3d_experiment trace-evaluate` freezes and evaluates one configured
best raw/EMA. `run_fiber3d_experiment comparison` executes the remaining serial
campaign from its control JSON: audit the completed initial raw evaluation,
evaluate S1a EMA, run JSON training/evaluations, run combined training/evaluations,
then select the largest complete-reference mean segment length. It aborts on any
failed training or incomplete evaluation, never skips a condition. The campaign
records comparison.json and per-evaluation selection.json, workload.json,
prepare/validate-full/full logs and trace_audit.json under its output root.
Continuation uses the original winning-stage config with2Mupdates and resumes
that run's current raw state, including EMA history. It does not substitute the
best EMA inference weights into an unrelated optimizer state.
