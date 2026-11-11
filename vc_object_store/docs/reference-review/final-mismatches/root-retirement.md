# FM-P01 — Track grower still submits numeric write guards

Baseline: HEAD `153c9a10f` plus working-tree patch SHA-256 `286b96ae90b5960043810003370f6ec3a66110597251dbcdd39477b313501235`. Older omission, already present at `0c7e4e521`; not introduced by the preceding PF fixes. Independently accepted by Astra in `astra-root-adjudication.md`.

`spiral-fitting/grow_track_graph.py::retire_surfaces` reads `current_version` and passes integer values directly to `db.write(expected_versions=...)`. The native transaction codec correctly rejects these obsolete guards. The shipped `next_task` path calls this helper for `--store` plus `--overwrite` when a grown surface already exists, so replacement fails before growing.

Isolated runtime counterexample: create a live surface and call `retire_surfaces(path, [uuid])`. It raises `ValueError: transaction expectations require a complete reference or null; select a reference before submission`. No retirement occurred. The empty-list case is not evidence of working live retirement.

Use the existing shared full-reference selection/expectation API, including checked treatment of missing identities. Do not restore numeric transaction compatibility. Other numeric discovery APIs explicitly converting in a native snapshot are not this defect.
