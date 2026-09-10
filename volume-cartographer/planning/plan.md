# VC3D render and fetch efficiency

Improve interactive remote-volume rendering by making fetch demand observable,
then using measured queue and throughput behavior to refine request priority and
adapt download concurrency to available bandwidth.

Work in measured phases:

1. Add compact, optional per-scale queue and throughput diagnostics to every
   slice view.
2. Record representative navigation traces and identify priority inversions,
   stale work, duplicate demand, and under/over-subscribed download periods.
3. Correct fetch ordering while preserving rendered values and deterministic
   request semantics.
4. Add bounded adaptive download concurrency driven by stable throughput and
   queue measurements, with conservative fallback behavior.
5. Benchmark each behavioral change separately and retain only measured wins
   without rendering regressions.
