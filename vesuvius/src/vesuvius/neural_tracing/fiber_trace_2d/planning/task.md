# Combined Direction And Contrastive Embedding Tracer

Add an optional Trace2CP/trace visualization mode that greedily minimizes a combined
direction plus embedding score at every tracer step.

For now the goal is inspection and score tuning through visualization, not replacing
the default tracer or changing training. Candidate steps should be a discrete angular
fan around the current oriented reference direction with a default maximum of 25
degrees. The initial default should be 1 degree spacing from -25 to +25 degrees,
giving 51 candidates; a coarser 2 degree spacing should be configurable.

At each step, score each valid candidate using even default weights for:

- direction agreement with the current model direction;
- embedding agreement with the previously accepted trace point;
- embedding agreement with the two enclosing Trace2CP control points;
- embedding agreement with all control-point embeddings from the same fiber.

The visualization should make it easy to compare the current direction-only trace
against the combined direction+embedding trace and see which gives the best
Trace2CP metric.
