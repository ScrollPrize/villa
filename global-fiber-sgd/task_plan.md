# Global fiber SGD: data reconnaissance

## Current task

Commit recommended H/V parameters and visualize separate H/V/broken layers;
current detailed plan [hv_vis_plan.md](hv_vis_plan.md).

Stage2 H/V assignment after normal alignment, with whole-fiber rejection:
[hv_stage_plan.md](hv_stage_plan.md), reviewed mathematical contract and matrix.

1000long retained-fiber visualization plus catalogue umbilicus; detailed current
plan in [vis_1000_plan.md](vis_1000_plan.md). Existing exporter, no new inference.

Signed-normal solution visualization with full plan in
[sign_visualization_plan.md](sign_visualization_plan.md). CPU-only OBJ/PLY export,
original whole AFV curves and exact oriented endpoint normals; no new inference.

## Follow-up: shared BP integration (2026-10-08)

User requests merging existing BP code and explaining its Lasagna constraint
extraction. Preserve AFV setup in a commit; merge latest shared implementation
from fiber-exp5, resolve conflicts by retaining both branches' features, attempt
native build and run available benchmark-support tests. Document extraction
versus solver behavior in bp_constraints.md. No AFV-to-BP adapter or SGD solver
is requested yet. Independent review covers merge resolution and the overview.
Spec update: no extraction semantics change. Docs/log update: merge provenance,
conflict resolution, validation limitations and shared entry points.

## Task

Set aside threshold/skeleton experiments. Start `global-fiber-sgd` from current
`origin/main`, download Paris4_v1.afv, inspect the existing AFV specification and
data, and identify the source scan. Future aim: global fiber assignment/resolution
using PyTorch SGD. Do not implement the optimizer in this initial inspection.

## Plan

1. Preserve dirty experimental work without carrying it onto the new branch.
2. Fetch main and create the branch; download the AFV into this working folder.
3. Reuse the existing AFV specification and inspect the database read-only.
4. Verify metadata, geometry counts, coordinate frame, database integrity and
   source-volume shape; record provenance and limitations.

## Spec update

No changes to tracing or AFV contracts. This is a new exploratory workspace;
native L0 XYZ coordinates remain authoritative. Fiber geometry does not prove
unique anatomical fiber identity or winding membership.

## Docs updates

README.md records the dataset, findings, source links, branch/stash provenance,
validation and future SGD scope. metadata.json preserves the source metadata.
task_log.md records this initial inspection; downloaded data stays out of Git.

## Validation

Check file size/SHA256, SQLite quick_check, complete flag, format IDs, metadata
versus row counts, block indexing/overlaps/finite points, and native Zarr shape.
No inference, model training or benchmark reruns are required.
