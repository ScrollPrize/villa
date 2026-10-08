# Capability-aware scheduling

User-authorized protocol v1 amendment, 2026-09-10. No GPU tests required.

Plan:
1. Add immutable registration flags `pausable` and `adaptable` (default true).
   Under full-release reconfiguration, non-pausable also requires non-adaptable.
2. Preserve non-pausable live grants. Non-adaptable tasks retain their first
   allocation across pauses. Reject controller/request/override changes that
   violate these promises. Voluntary owner release remains possible.
3. Order all candidates by explicit priority, then registration order. Exclusive
   requests have no priority bonus. Wait for non-pausable blockers without
   unnecessarily pausing others; expose a reason through polling/status.
4. Test synthetic ownership transitions, recovery, priority, capability validation,
   controller overrides and wire/client behavior; document reference usage.

Status: implemented. Flags live in registration/schema/client; policy and engine
enforce fixed/non-preemptible grants and explicit-priority admission. Ownership
barriers cover both entering and leaving exclusivity. The authorized protocol
amendment, recovery behavior and client usage are documented.

Validation (synthetic resources, no GPUs):
`GPU_MANAGER_SCHEMA_PYTHON=/usr/bin/python3 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest -q gpu_manager/tests`

80 tests passed in 12.59s, including 19 new scheduling checks. Unix socket/process
tests require execution outside the sandbox; system Python supplies the already
installed JSON Schema validator. No installs or production processes changed.
