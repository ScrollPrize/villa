# Task: Infer signed integer winding labels in crop BP

Extend the crop H/V belief-propagation path with winding-number inference.
Before constraint inference, align regular Lasagna normal axes over the trace
crop using the shared BP normal-alignment implementation. Use those aligned
normals to orient every usable perpendicular constraint: for ordered pieces
`A -> B`, positive signed winding means `winding(B) - winding(A) > 0`.

Infer winding in two stages:

1. solve a gauge-fixed continuous weighted difference relaxation;
2. center small integer candidate sets on that solution and run categorical
   sum-product BP, expanding candidates adaptively when posterior mass or the
   MAP label reaches a candidate boundary.

Parallel evidence favors equal winding labels. Perpendicular evidence favors
the measured signed winding difference. Same-trace continuity is exact zero
difference. Fix one deterministic crop-central piece per connected component to
winding zero. Report integer MAP labels, posterior means and confidence, while
retaining the continuous solution as a diagnostic. H/V and winding inference
share topology and output but remain mathematically factorized.
