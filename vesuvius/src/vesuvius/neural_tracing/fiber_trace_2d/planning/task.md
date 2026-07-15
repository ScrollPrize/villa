# 3D Fiber PyTorch DataLoader Parallelism

The previous 3D loader optimization did not fulfill the requested runtime
parallelism requirement. It replaced the slow zarr-crop/grid-sample path with
VC3D coordinate sampling, but the training overlap path used a
`ThreadPoolExecutor` around `load_batch` instead of PyTorch data-loader
parallelism.

Required behavior:

- Replace the thread-backed `_OrderedBatchLoadPipeline` with a real
  `torch.utils.data.DataLoader` based batch-loading path for
  `fiber_trace_3d.train`.
- Use PyTorch worker processes (`num_workers`) for independent batch loading,
  not Python threads for CPU-heavy work.
- Each worker must lazily construct/open its own `FiberTrace3DLoader` and VC3D
  sampler state inside the worker process; opened zarr/VC3D handles must not be
  pickled from the main process.
- Preserve deterministic training order exactly. Changing worker count,
  prefetch factor, batch size, or max steps must not reshuffle the sample-index
  sequence consumed by training.
- DataLoader items should represent whole training batches or otherwise avoid
  nested/default collation that would corrupt `FiberTrace3DBatch` dataclasses.
- Workers must return CPU tensors only. Any optional worker-side CUDA
  coordinate generation must use worker-owned CUDA context/stream and must
  synchronize/copy to CPU before returning data to the main training process.
- Main training process moves the returned batch to the training device before
  forward/backward.
- `--benchmark --load-only` must exercise the same DataLoader path so loader
  parallelism can be measured directly without model work.
- Keep the existing VC3D coordinate sampling and search-free segment target
  generation semantics from the previous patch.
- Document and test this explicitly. Silent fallback to the old thread-backed
  path is not acceptable.
