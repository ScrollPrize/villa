# Task: TensorStore whole-volume inference prefetch

Replace the shared Fiber/Lasagna whole-volume inference input reader with a
TensorStore-backed asynchronous bounding-box reader and decouple read-ahead
capacity from GPU/result slots.

Requirements:

- both Fiber and Lasagna must use the same implementation in
  `lasagna.tiled_predict3d`;
- TensorStore must asynchronously read/decode upcoming tiles while GPUs infer;
- tile coordinates remain lazily generated rather than materialized globally;
- prefetch depth, TensorStore I/O concurrency, and cache memory are bounded and
  configurable independently of GPU result slots;
- preserve exact input slicing, dtype conversion, reflect-border padding,
  canonical commit order, resume semantics, and numerical output;
- retain a Python-Zarr fallback for diagnosis/portability;
- measure the reader and end-to-end pipeline before/after on controlled input
  and leave representative eight-GPU measurement hooks in normal logging.
