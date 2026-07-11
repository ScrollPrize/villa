# Trace2CP Metric For Test Evaluation

User request:

- Make Trace2CP an actual metric calculation function, not only a
  visualization score.
- Change Trace2CP error to vertical error per horizontal step/span.
- Use the actual closest trace-to-trace point for the metric, not the
  center-focus optimized/penalized point.
- When called on a set of test CPs, each CP uses the segment to the next CP for
  metric calculation.
- The metric path does not need visualization or full fused/refined trace
  building.
- Add the metric to training test runs and select the best model by this
  metric when test data is configured.
- Adapt Trace2CP visualization commands to also print/output this metric.
- The metric is averaged over all valid evaluated segments.
- If traces do not overlap before hitting the valid/RF edge, use the default
  maximum error: vertical distance from centerline/CP y to validity edge
  divided by the segment width, because exact early/late edge intersection is
  considered noise for now.
