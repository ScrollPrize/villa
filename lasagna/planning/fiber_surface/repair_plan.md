# Local patch reconstruction experiment

Follow-up: user requests larger patches first. Compare [16,8,4,2] halos with
largest connected patch first within each halo against completed [2,4,8].
Keep all other settings/inputs and acceptance checks identical; neither run
promoted to default. No truncated subset or extra fiber rejection.

Large-first alone barely improves immediate fold counts. Follow-up: preserve
incoming boundary slope by penalizing Laplacian changes on frozen boundary
vertices (weight10). Evaluate full/.5/.25 moves toward reconstruction; all
retain the same hard acceptance checks. Compare against small-first/large-first
including matched guarded continuation controls. No removal of difficult patches.

If continuation erodes the gains, test guard10000 vs100, keeping reconstruction
and all other parameters unchanged. Retain both raw repair and continued meshes.

Task: repair both saved 3x-smoothness surfaces, then continue joint UV/XYZ fitting.
Preserve inputs and all previous results; keep H selection, immutable XYZ and
initial UV displacement reference. Use shared non-exclusive GPU grants on 0–5.

1. Detect local opposing triangles, near-collapse, and neighboring quad normal
   reversals. Grow connected UV-grid patches with halos 2, 4, 8 cells. Freeze
   vertices outside each patch, including its boundary and outer mesh boundary.
2. Reconstruct patch interior by sparse least-squares harmonic interpolation with
   fixed boundary and soft full-XYZ bilinear fiber guides at saved final UV.
   Try guide weights 0, 0.1, 1. UV cannot move during reconstruction.
3. Accept only finite candidates reducing local defects, introducing no new
   defects in previously healthy affected cells/edges, and satisfying local
   and global attachment-fit guards. Otherwise enlarge/retry, retaining original
   geometry when repair fails. Report attempts, rejects and remaining defects.
4. Resume existing joint optimizer from repaired XYZ and saved attachment UV,
   keeping displacement measured from ORIGINAL UV. Fresh optimizer state (not
   an exact checkpoint resume). Add opt-in orientation/area guard against the
   repaired healthy geometry. Also run matched no-repair continuation control.
   Preserve before/repaired/continued meshes and metrics, even on regression.
5. Test synthetic fold recovery, frozen boundary and UV, failed-repair safety,
   guard gradients, checkpoint reconstruction and existing regression suite.
   Run both original and deformed regional datasets through approved runner.

## Spec update

Opt-in local repair and continuation only; no default promotion. Local fold
diagnostics are not global self-intersection certification. Curved/bad boundary
or incompatible UV constraints can prevent repair; report, do not hide failures.

## Documentation and changelog

Record exact configs, commands, fit/geometry before/after, runtime, accepted
patches and limitations in repair_results.md, docs, changelog and active task_log.
