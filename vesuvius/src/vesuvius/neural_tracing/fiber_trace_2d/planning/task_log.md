# Embedding/manager integration — 2026-09-11

- User explicitly requests merge of fiber-training-embedding and manager review, with tests/profiling exclusively managed to prevent contamination.
- Current branch fiber-training-retrain at fed6dc24a; incoming15b4005db. Two merge bases; use normal merge, no history rewrite. Local dirty changes are known inference/manager/bbox task work and will be checkpointed by subproject.
- Incoming contains bootstrap/socket/idle lifecycle and logical-CPU changes, embeddings/training updates, and an independently added exclusive.py/test_exclusive.py that overlap our safer watchdog launcher. Explicit semantic conflict resolution required.
- Previous bbox task archived in docs/unified_bbox_reads_task_log_20260911.md. Last runtime inspection found shared extension reverted to old API; must revalidate before benchmarking. No journal edits or unmanaged GPU runs.
