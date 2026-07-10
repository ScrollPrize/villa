# Training Batch Config Validation, Prep Slowdown, And Pipeline Depth

Remove the warning:

`fiber_trace_2d train: patch batch is ..., expected 64 for the default 4 control points x 16 strip offsets`

Non-default patch counts are valid and should not print anything. If the
training CP count and configured loader batch size disagree, fail early with a
clear config error instead of silently ignoring one of the settings.

Also fix the observed training slowdown after the parallel preparation change:
benchmark/profile shows CUDA value augmentation blur running as a Python loop
over patches, creating many tiny CUDA convolutions per batch.

The latest training output still shows large `wait_ms` spikes, and the active
example config does not specify `pipeline_depth` or `pipeline_workers`, so it
falls back to a shallow two-batch pipeline. Make the training defaults and
example config actually keep several batch loaders in flight, and print the
effective pipeline settings at startup.
