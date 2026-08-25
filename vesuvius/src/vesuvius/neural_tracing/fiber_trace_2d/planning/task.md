# Task: continuous deterministic crop tracing and zero-copy graph access

Accelerate `vc_fiber_trace_chunk` without changing its numerical tracing or
strongest-first coverage result.

- Replace synchronized candidate batches with continuously fed workers.
- Candidate computation may run speculatively, but completed candidates must
  finalize strictly in seed order so coverage and accepted output remain
  deterministic.
- Replace repeated immutable-graph map traversal and vector copies with stable
  indexed storage and borrowed directional views.
- Make the view ownership model usable by cache-backed graph sources through a
  single lease per returned view, never reference tracking per element or
  search state.
