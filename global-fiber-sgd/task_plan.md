# Global fiber SGD: data reconnaissance

Current: bbox_gt_phase_fix_plan.md; independent review approved.

Current: bbox_gt_eval_plan.md (expanded trusted GT and cross-constraint scoring).

Current: bbox_gt_plan.md, independently reviewed.

Current: uv_loss_search_plan.md, adaptive finite-grid final-L1 weight search.

Current: uv_bucket_lr_plan.md, fresh state10x thenselectedbasewidthLR20/.2.

Current: uv_bucket_split_plan.md.

Current: uv_lr_default_plan.md, adopt measured high LR then discuss disentanglement.

Current: uv_class_balance_plan.md; supersedes prior completed regional trials.

Current: uv_reference_subset_plan.md. Older entries below are historical.

Current: uv_diagnostics_100_plan.md (independently approved).

Current: uv_distributed_plan.md, larger batches then true multi-GPU solve.

Current: uv_coordinate_plan.md.

Current: uv_lr16k_plan.md.

Current: uv_local_window_plan.md.

Current: uv_squared_vertical_split_plan.md.

Current: uv_vertical_split_plan.md, independently approved.

Current: uv_remove_pair_phase_plan.md, independently approved.

Current: uv_distance_objectives_plan.md, independently approved.

Current: uv_pair_fusion_plan.md, independently reviewed.

Current: uv_global_accumulation_plan.md, independently reviewed.

Current: reference_init_vis_plan.md, independently reviewed.

Current: uv_reference_init_plan.md and reference_affine_v_plan.md, independently
reviewed. These supersede the pointwise physical-Z visualization below.

Current: reference_physical_z_plan.md, independently reviewed.

Current: uv_broken_joint_plan.md, independently reviewed.

Current: uv_fullpass151_vis_plan.md, independently reviewed.

Current plan: uv_full_pass_plan.md (independently reviewed). Historical plans below
are not the current execution specification.

Current task: uv_joint151_strong_plan.md. One joint solve, no stages.

Current task: uv_fresh151_plan.md (fresh joint solve and within-run stages).

Current task: uv_151_palette_plan.md.

Current correction: reference_comparison_plan.md TIFF-only section, reviewed.

Current task: [reference comparison](reference_comparison_plan.md), independently
reviewed. Original AFV fibers share one median assignment across solved pieces.

Current task: [reference acquisition/proposal](reference_surface_plan.md).

Current task: [100-sweep density](fiber_density100_plan.md).

Current task: [100-sweep maps](uv_visualization100_plan.md).

Current task: [H/V rejection diagnostics](hv_rejection_plan.md).

Current task: [kollesis candidate diagnostic maps](kollesis_plan.md). Four
non-endpoint metrics over U and U/V, read-only saved20solutions, CPU-only.

Current task: [saved U/V visualization](uv_visualization_plan.md). Optimizer
state committed c01fb1ecd first; no further optimization in this task.

Current task: [aggregated categorical targets](uv_categorical_plan.md).
The task references below describe previous experiments.

Current task: [fixed sign charge](uv_sign_charge_plan.md), superseding the
completed shared-anchor integration below. Target zero retained sign violations.

Current task: uv_anchor_plan.md; user has resumed global layout after completed
direction stages. This supersedes the historical paused-layout notes below.

## Current task

V-fiber consistent up-direction alignment and breakage: task.md and
[v_direction_plan.md](v_direction_plan.md). H state committed3f5b443e8 and frozen.
Global U/V remains paused.

## Prior H task

H-fiber direction breakage and visualization: task.md and
[h_direction_breakage_plan.md](h_direction_breakage_plan.md), independently reviewed.
Global U/V remains paused. Reuse existing source/tangent cache and rendering.

## Earlier task plans

H-fiber circulation alignment: task.md and h_direction_plan.md. Global U/V is
PAUSED by user request; results/notes preserved in uv_layout_task_log.md.

Global probabilistic U/V layout: task.md and uv_layout_plan.md.

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
