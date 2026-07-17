# Trace2CP Refined Strip Off-Strip Clipping

Native 3D Trace2CP refined/regenerated strip visualization currently fails
when a traced point leaves the source strip valid area:

`ValueError: refined Trace2CP trace leaves the source strip valid area`

This should not be fatal for visualization. Off-strip trace points, including
original trace endpoints, should be clipped/ignored while building the refined
strip. Strict errors should remain for malformed traces, non-finite values, or
cases where too few valid points remain to build a refined strip.
