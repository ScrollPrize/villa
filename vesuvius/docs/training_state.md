# Shared snapshots and in-RAM training handoff

`fiber_trace_3d.training_state` is the existing state-handling library extracted
unchanged from `3a4a7b2c1`. Ordinary snapshots use its common model, optimizer
and AMP-scaler payload/restore helpers. This does not introduce another trainer
or enable cooperative pause/reallocation by itself.

## Ordinary disk snapshots

`_save_snapshot` still unwraps DDP/DataParallel. Its keys remain `model`,
`optimizer`, `step`, `config`, `metric`, `metric_name` and, only with an enabled
scaler, `grad_scaler`. Model keys, tensor values and JSON-safe configuration
conversion are unchanged; serialized bytes/order are not a compatibility
contract. Handoff progress and RNG are not silently added to disk checkpoints.

`_load_snapshot` still accepts a checkpoint or raw model state dictionary,
loads weights strictly and returns the saved step (zero for raw weights).
Optimizer/scaler restoration is optional. Explicit current optimizer
hyperparameters are reapplied after loading optimizer state; without that
payload they are not reapplied. Disabled scalers are not saved/restored.
Disk resume intentionally retains its existing no-RNG-restoration contract.

`model_optimizer_payload` returns state-dictionary views for immediate
serialization, **not** an immutable snapshot. Use `cpu_clone` or
`capture_handoff` to retain independently owned state while training continues.

## Completed-boundary RAM handoff

`TrainingBoundary(step, done, capture, phase="update")` describes a completed
logical update/test unit, not a physical microbatch. Its callback allows the
caller to capture only when needed.

```python
from vesuvius.neural_tracing.fiber_trace_3d.training_state import (
    capture_handoff, restore_handoff,
)

# Finish the optimizer/test unit and drain asynchronous preparation first.
state = capture_handoff(
    model=raw_model, optimizer=optimizer, grad_scaler=scaler,
    progress=progress, cuda_device=device,
)
# Keep state on CPU while replacing authorized GPU workers.
progress = restore_handoff(
    state, model=replacement_model, optimizer=replacement_optimizer,
    grad_scaler=replacement_scaler, cuda_device=replacement_device,
)
```

The payload owns detached CPU copies of model tensors/buffers, optimizer,
enabled scaler and caller-supplied progress, plus Python, NumPy and Torch CPU
RNG state. CUDA RNG is accessed only with an explicit worker device; CPU
callers never probe or initialize CUDA. A saved CUDA RNG can be restored on a
different granted device. GPU state without a device, or GPU restoration
without CUDA RNG, raises. Callers must supply compatible state/devices;
malformed restores are not transactional.

Progress is application-owned: include canonical sample frontier, completed
update, best metrics, pending test rows/group cursor and logging progress.
Never reconstruct them from rank-local microbatch counts. Restoration clones
saved optimizer state and returns an owned copy of progress, so subsequent
CPU optimizer updates cannot mutate the retained handoff. Uncommitted gradients,
queued samples, dataloaders, processes and manager allocations are not captured.

`cpu_clone` supports Torch state, dictionaries (preserving OrderedDict state
metadata), lists, tuples, NumPy arrays/scalars and primitive values. Unsupported
objects raise instead of being silently shared. Copy/hash cost is proportional
to state size and paid only when a caller requests a boundary handoff.

## Identity checks

`config_fingerprint` hashes a JSON-compatible numerical/data configuration
with sorted keys, excluding only `training.cooperative`. It neither mutates
the config nor accepts nonfinite JSON numbers.

`replica_signature` hashes model, optimizer, scaler and progress, excluding the
top-level rank-specific RNG entry. It requires dense strided CPU tensors and
supported Python containers; CUDA tensors and arbitrary objects are rejected.
This is a boundary consistency check, not a persisted protocol or checkpoint
checksum. NumPy containers supported by cloning are not signature inputs.

## Validation

```bash
cd vesuvius
python -m pytest -q tests/neural_tracing/test_training_state.py -k 'not cuda'
```

Standalone source tests check RNG/progress restoration and owned state.
Real cached Paris4 CT/annotation tests use a small sample-independent
seven-channel head to compare subsequent loss, model and AdamW state exactly
after RAM handoff or disk save/load, including CPU BF16 and
enabled/disabled scalers. They also cover raw weights, LR/WD overrides, schema,
wrapper keys and CPU-only behavior. Source tests depending on later
`training_units`/physical-test-batch integration remain deferred with that
trainer extraction; these utility tests do not claim manager interruption
coverage.

CUDA tests must be launched under a GPU-manager reservation for two GPUs:

```bash
FIBER_ASYNC_CUDA_TEST=1 python -m pytest -q \
  tests/neural_tracing/test_training_state.py -k cuda
```

The reserving caller sets `CUDA_VISIBLE_DEVICES` and
`FIBER_COOPERATIVE_TEST_GPUS` before launching. Tests cover FP32, BF16 and scaled
FP16, with restoration to the original or replacement device. Exact equivalence
is checked on tested hardware/operations, not promised across different GPU
architectures or kernels. No data/checkpoints are downloaded or modified and
no precision defaults change.
