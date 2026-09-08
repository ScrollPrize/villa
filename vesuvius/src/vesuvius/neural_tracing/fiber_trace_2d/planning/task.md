# Shared Fiber3D sample-worker pool

Replace the reusable benchmark pipeline's fixed loading and augmentation
process pools with one shared pool capped at 128 processes. Free workers must
take runnable work from either stage without dependency inversion: augmentation
is eligible only after its matching load completes, and loading must continue
whenever no augmentation is runnable.

Use gradual bounded admission instead of immediately exposing the full
lookahead. Start with a small window and, for every completed load, advance the
admission frontier by two until the configured maximum lookahead is reached.
Keep deterministic ordered delivery, bounded shared-memory slots, explicit
failure handling, and unchanged sample values. Benchmark CPU-only against the
committed 229.75 samples/s production-throughput reference, using no more than
128 processes/CPU threads. Do not use GPUs.
