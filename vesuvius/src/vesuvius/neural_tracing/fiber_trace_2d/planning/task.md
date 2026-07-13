# Torch-Vectorized Trace2CP DP Backend

- Implement a torch-vectorized backend for the Trace2CP monotone DP optimizer.
- Vectorize as much of the per-column work as possible over layers, rows,
  moves, and sampled transition columns; keep only the DP column recurrence
  sequential.
- Prefer parallel tensor work over dynamic Python loops, including processing
  all z layers together.
- Keep the existing NumPy/Python implementation as a fallback for direct tests
  and non-torch calls.
- Preserve current side/z/top Trace2CP semantics, progress output, and final
  timing rows.
