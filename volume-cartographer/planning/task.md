# Task: separate source acquisition from global I/O configuration

Correct the shared chunk-cache APIs so acquiring a volume source has no
service-wide scheduling side effects and I/O concurrency is configured only on
the `ChunkCacheService`.

- Replace the mixed `ChunkCacheOptions` contract at the service source factory
  with source-only acquisition settings.
- Rename the source factory to describe its acquire-or-reuse behavior.
- Remove the volume-scoped I/O-thread setter; regular concurrency is one
  explicit service-global setting.
- Reconfigure the service's existing source-read scheduler in place. Running
  and queued work must remain attached to that scheduler and each chunk request
  must execute at most once because of a configuration change.
- Preserve the intended VC3D adaptive default, isolated batch-cache behavior,
  decoded contents, priorities, demand, listeners, and adaptive state.
- Replace the scheduler-migration regression test that currently expects a
  duplicate fetch and causes the non-Valgrind CI failure.
