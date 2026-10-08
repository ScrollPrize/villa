# Task log

- Started unified loading/calibration task; earlier uncommitted profiling work is
  preserved. Managed training already shares worker factories/queue engine, but
  duplicates construction; managed evaluation bypasses it entirely.
- Baseline: 512 deterministic test patches, 8 GPUs, physical batch16:
  synchronous cold19.2s/warm12.1s; legacy persistent DataLoader cold29.5s/warm3.17s.
  Initial model/loader/distributed setup approximately25s, measured separately.
