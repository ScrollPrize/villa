# Task: share canonical anchors across fiberlet quantization

Extract, fit, filter, and persist canonical float anchors once. Every position
or fitted-direction quantization scenario must reuse that anchor dataset and
derive its endpoint view without rerunning anchor extraction. Only fiberlet
candidate generation, sampling, DP, routes, and prefixes remain specific to a
position/direction geometry scenario.

Preserve strict cache identity and reuse of existing fiberlet caches. Validate
the float-position plus compact-direction plus float-cost scenario and the Q4
position plus float-direction plus float-cost scenario.

Add and validate float-position plus compact-direction scenarios with `uint8`
and `uint16` replay costs. Both must reuse the existing compact-axis geometry
cache namespace rather than create cost-specific geometry. Because that cache
is populated on demand, computing a stable per-owner cost range may complete
missing compact-axis chunks in place. Run both over the full radius-768
corridor.
