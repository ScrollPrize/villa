# Task: measure remote bandwidth from received HTTP bytes

Replace the completion-span bandwidth estimate used by VC3D and adaptive
download admission. HTTP Zarr downloads must report response-body bytes while
they arrive, producing one service-wide aggregate bandwidth measurement.

- The status bar and adaptive controller must use the same estimator.
- Idle gaps between download bursts must not enter the denominator.
- Adaptive epochs require at least five active seconds and at least the current
  admission count of successful completions.
- Chunk completion remains the source of latency and success/failure data.
- Non-streaming/custom fetchers may fall back to mean individual transfer rate
  multiplied by the admission used for those samples.
