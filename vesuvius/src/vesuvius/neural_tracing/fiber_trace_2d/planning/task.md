# Task: two-stage regional Fiberlet graph reduction

Extend `vc_fiberlets chunk-route-stats` from one analysis box to a cubic
base-coordinate region containing a regular grid of analysis boxes. With a
256-base chunk and a 512-base region, stage one analyzes the eight globally
aligned `256^3` boxes intersecting the region and persists each box's retained
Fiberlets in a global reusable reduced cache without modifying the original
graph cache. The requested region selects cache chunks; it must not define a
region-specific cache identity or fixed coverage manifest.

Add an explicit two-stage mode. Stage two reads the stage-one reduced graph and
analyzes the offset grid of equal-size boxes whose minima are shifted by half a
chunk on all three axes and which remain fully inside the region. Thus a
512-base region with 256-base chunks has one centered stage-two box at
`region_minimum + (128,128,128)`.

Report stage-two retained and removed Fiberlets against the original input
Fiberlet population in the same stage-two crop. Report internal Fiberlets
separately so boundary entry/exit edges do not hide the reduction. Reuse the
existing Fiberlet dataset, serialization, chunk cache, decoded LRU, graph, and
exact route-analysis infrastructure. Do not introduce compatibility handling
for this unpublished command or mutate the authoritative source cache.
