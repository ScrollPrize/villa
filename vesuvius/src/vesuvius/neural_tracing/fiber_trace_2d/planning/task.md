# Trace2CP Similarity Debug Column

Add embedding-similarity debugging to Trace2CP visualization.

The single-pair tracer visualization should show an additional column when
embedding outputs are available. The column should render:

- similarity to the start CP embedding;
- similarity to the target CP embedding;
- same-fiber global CP-bank similarity when the combined Trace2CP fiber bank is
  available;
- similarity to the final sampled embedding of the forward directional trace;
- similarity to the final sampled embedding of the reverse directional trace.

This is a debug visualization only. It must not change Trace2CP tracing,
metric, or refinement semantics.
