# 128 sd2 Single-Output 3D Fiber Config Task Log

## Planning Notes

- Checked `fiber_trace_3d/model.py`: `conditioned_decoder_enabled` defaults to
  false, and with one branch the model emits seven sigmoid channels.
- Checked `fiber_trace_3d/train.py`: `_forward_loss_tensors(...)` calls
  `compute_conditioned_losses(...)` only when the model advertises conditioned
  decoder mode; otherwise it calls the legacy `compute_losses(...)` path.
- The requested old single-output supervision is therefore still supported
  without runtime code changes.

## Deviations / Deferred Items

- Independent agent review of the small config plan was not run because no
  separate review agent was explicitly authorized in this turn.
- No production code changed; this task only updates a training config and
  stale planning/spec text.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py`
  passed.
- A `PYTHONPATH=vesuvius/src:.` smoke script loaded
  `train_s1a_nml_all_128_sd2.json`, built the model, and verified:
  `conditioned_decoder_enabled=False`, `output_channels=7`, and
  `direction_branch_count=1`.
- The same smoke script used a dummy non-conditioned model with
  `_forward_loss_tensors(...)` and confirmed the legacy loss keys were
  produced: `total`, `direction`, `presence`, `angle_mean_deg`,
  `branch0_fraction`, `branch1_fraction`, and `selected_score_mean`.
- The smoke check intentionally did not run a real 128-cube training forward on
  CPU.
