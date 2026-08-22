# Task: separate fiber replay progress

Replace the misleading composite fiber replay progress percentage with explicit
cache/preprocessing and reference-tracing progress. The tracing value must be
the actual minimum reference-arc fraction reached by the greedy and fiberlet
tracers. Keep cache generation, prefetching, tracing, and output behavior
unchanged.

When visualization or publication continues after tracing, report it as output
work rather than folding it into either cache or trace progress.

Show `elapsed` only once on the active compact line, and remove the cache/prep
field as soon as its scheduled fraction reaches 100%.

For trace progress, additionally show an ETA based on recent progress speed,
the number of search states expanded by the latest bounded lookahead decision,
and the minimum local loss-per-prediction-voxel cutoff that actually stopped
candidate expansion at its maximum-lookahead front.
