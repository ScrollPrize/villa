# Pausable demand-driven sweep GPU ownership

Stop the current sweep. Keep CPU tracing running while GPU work is paused.
Release GPU workers/reservation after 30 seconds without tile work and request
again on demand. Reuse the locked manager protocol and shared inference.
Start released: request no GPUs until the first input tile is ready.

Goal continuation: improve embedding tracing by completing the previously
requested method refinements. Reuse cached predictions; for coverage missing
from cache, use the requested adaptive manager path, starting without a grant,
pausing cooperatively and releasing after30s idle. Preserve completed searches.
