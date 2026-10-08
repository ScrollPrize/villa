# Global fiber SGD: data reconnaissance

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
