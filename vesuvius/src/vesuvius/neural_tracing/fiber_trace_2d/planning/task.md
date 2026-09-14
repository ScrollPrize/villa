# Pausable demand-driven sweep GPU ownership

Stop the current sweep. Keep CPU tracing running while GPU work is paused.
Release GPU workers/reservation after 30 seconds without tile work and request
again on demand. Reuse the locked manager protocol and shared inference.

Goal continuation: improve embedding tracing by completing the previously
requested method refinements. While keeping GPU work stopped, use existing
predictions for CPU-only replay of remaining families and exact repeats.
Do not convert missing cached coverage into new GPU inference.
