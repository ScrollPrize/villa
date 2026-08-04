# Task: process-parallel native product accumulation

Implement the shared Lasagna/Fiber accumulator as a deterministic process
pipeline and add one portable native in-place add extension with runtime
AVX-512/F16C acceleration for float16 mmap storage. Preserve chunk ownership,
rolling-ring safety, canonical tile ordering per chunk, bounded shared-memory
slots, asynchronous flush, and float32 weights/flush arithmetic. Make the
parallel/native solution useful for both float16 and float32 accumulators.
