# Native Fiber Trace Locality And Scheduling Optimization

Continue optimizing the retained native precomputed C++ fiber tracer. The
original cap-32 baseline was:

- 1.869s wall / 8.222s CPU
- 6,910,839 candidates and 4,318 generations
- 7 restarts over 87 segments

After result-neutral and accepted numeric-relaxed optimization, the new
retained baseline is plain cap 32:

- 1.161s wall / 4.945s CPU
- 6,910,839 candidates and 4,318 generations
- 7 restarts over 87 segments

Preserve 7 restarts. Result-neutral scheduling, storage, and locality
changes must preserve deterministic output exactly. Search approximations may
be tested separately but must not replace the retained default unless measured
quality retains the 7-restart baseline.

Investigate and test these proposals independently:

1. Reduce small-batch thread-pool scheduling overhead.
2. Select the capped parent prefix without fully sorting all parents.
3. Store only evaluated capped-frontier children while preserving original
   global indices for deterministic ties and reconstruction.
4. Spatially order candidate sampling by chunk and integer voxel cube, then
   scatter scores back to original candidate indices.
5. Gather each unique integer voxel cube once and reuse its ordered corners for
   candidates with different interpolation fractions.
6. Keep a shared pinned sampling session across both lookahead depths, while
   retaining the mandatory decision barrier between them.
7. Measure bounded depth-two envelope prefetch and a small rolling pin window
   across consecutive trace steps.
8. Test fixed parent caps 28, 24, and 20 around the observed quality boundary.
9. Test adaptive cap escalation only after the fixed and result-neutral options:
   begin with a smaller cap and retry uncertain or failed work at cap 32/64.

Before every representative benchmark, sample host load. Run directly when the
host is quiet; if significant competing CPU work is active, wait for the user
to confirm resources are available. Reuse the exact approved command and cache
path.
