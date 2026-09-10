# Cooperative GPU manager

This is a reusable Python, single-node Unix-socket client/server package. It
coordinates promises, not hard resource enforcement. Follow the repository's
read-only startup/install policy and use the README for commands.

## Locked protocol

Protocol v1 was locked on 2026-09-10 after conformance and Fiber reference tests.
`docs/protocol-v1.md` and `src/villa_gpu_manager/schema.py` are authoritative.
Do not change wire fields/types, units, operations, capabilities, framing,
versioning, ownership/recovery rules or allocation semantics without explicit
user instructions authorizing a protocol change. This includes additive changes.
A version bump is not a workaround for that requirement.

Internal fixes may preserve the protocol, but require regression tests proving
that ownership cannot be released before the client's resources are gone.
Keep the package independent of Torch/VC3D/Lasagna. Linux/macOS support must be
guarded where peer credentials/process inspection differ; tests must not allocate
real GPUs unless explicitly authorized. No sudo or hard resource enforcement.

Normal tests must finish within minutes. The reusable cooperative test launcher
has a five-minute total budget; do not restore the long timing sweep without an
explicit request. GPU tests retain the currently authorized per-device limits.
