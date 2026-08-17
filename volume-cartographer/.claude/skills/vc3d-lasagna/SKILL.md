---
name: vc3d-lasagna
description: Operate VC3D's Lasagna fit service and optimization panel through MCP: distinguish datasets from manifests, ensure service readiness, submit/monitor/cancel optimizations, select outputs, repeat runs, and attach regular or fiber-inference manifests. Load before vc3d_lasagna_* calls.
---

# Lasagna service and optimization

Assume `vc3d-bridge-session`. Do not confuse two resources:

- A manifest-backed `lasagna` dataset is resolved by fiber tracing and Atlas
  for the current volume.
- The Lasagna fit service powers `vc3d_lasagna_*` optimization calls and has
  its own dataset and job lists.

One does not prove the other is available.

`scripts/spiral/flatten_spiral_checkpoint.py` is a third, separate workflow:
it owns a private ephemeral Lasagna service and is not observable through the
`vc3d_lasagna_*` calls. Use `vc3d-spiral-checkpoint-flattening` for it.

## Attach manifests

Use `vc3d_attach_lasagna_manifest(location, role, select=...)` for local,
HTTPS, or S3 manifests. Set `role` to `regular` or `fiber_inference` from the
manifest's actual purpose. The attachment is transactional, remote chunks stay
lazy, and ambiguous coordinate provenance must be reported rather than guessed.

For Open Data, select compatible representation refs through
`vc3d-open-data` instead of hard-coding a sample or manifest URL.

## Establish service readiness

1. Call `vc3d_lasagna_service_status` before starting anything.
2. If a service is required and absent, call `vc3d_lasagna_ensure_service`:
   omit host and port for an owned internal service, or supply both for an
   external service. Never supply only one.
3. Re-read status and record running/external, host, port, and `lastError`.
4. Call `vc3d_lasagna_list_datasets` and `vc3d_lasagna_jobs`; these query the
   service and can fail independently of service startup.

Do not start or restart a service merely to test status. Starting an internal
service is an explicit workflow mutation and may require the documented Python
interpreter.

## Optimize

1. Open a volume package and ensure the Lasagna panel/workspace is available.
2. Choose `reoptimize`, `new_model`, `offset`, or `atlas`. Resolve an actual
   config path; for atlas mode also resolve an atlas path unless already
   selected in the panel.
3. Pass an optional seed in L0 volume coordinates; VC3D rounds it to integers.
4. Call `vc3d_lasagna_start_optimization(..., wait=true)` and monitor the
   `lasagna` job. Only one bridge-submitted Lasagna job may run at a time.
5. Compare the bridge terminal record with `vc3d_lasagna_jobs`. Preserve
   service ids, console output, and failures.
6. Use `vc3d_lasagna_select_output` only with a returned/listed segment name,
   then verify through the segment lifecycle.

`vc3d_lasagna_repeat_last` inherits persistent panel state; use it only when
that exact state is known. Cancellation accepts a bridge job id, raw service
id, or the current bridge job when omitted; confirm terminal state afterward.

Report separately whether the manifest dataset, fit service, submission,
optimization, and output selection each succeeded.
