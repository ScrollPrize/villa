# Raw40500 historical1024 crop replay

Same frozen raw40500 manifest and tuning as the BP pilot, 26 tagged references clipped to the historical 1024³ crop. Exact replay case identities and lengths validated against the September23 beam replay. Cached predictions; no new inference or native changes.128 requested CPU threads, RelWithDebInfo, one run each.

Score is100/(ordinary failures+1). Filtered stops are separate, not ordinary errors. Full-prefix512/no-history uses the shared replay filter, including terminal endpoints; GT controls evaluation termination and reseeding. This is not a GT-free deployment score. No short-piece filtering.

| Policy | Replay % ↑ | Ordinary failures | Filtered stops | Retained % | Reverse checks | Wall s |
|---|---:|---:|---:|---:|---:|---:|
| unfiltered | 33.3333% | 2 | 0 | 100.0000% | 0 | 1.71 |
| prefix512 | 100.0000% | 0 | 6 | 94.8385% | 108 | 4.11 |

Approved runner: set control action `reviewed_bp_replay`, then `bash /home/hendrik/villa3/scripts/run_training_failure_mining`. Configs, commands, hashes, logs and outputs: `/ephemeral/hendrik/fiber_crop_bench/20261002_raw40500_prefix512_v1`. See adjacent JSON for complete native summaries.
