# Global U/V layout: first experiments

All earlier rejections fixed; baked H/V. Full-graph readout, no ground-truth layout labels.
Ordering errors use retained known-sign events; each run records its tolerance and broken readout below. Finite penalty is not a hard feasibility guarantee.
Rejection and arc retention use input pieces/arc after earlier stages. Distance errors are winding units.

| Run | Rejected pieces | Retained arc | Retained events | Sign errors | Mean distance error | Ambiguous (<90%) | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
|ccw_sweep10 initial|19.2202%|66.4410%|41.2562%|0.0002%|0.4115|80.8454%|794.13|
|ccw_sweep10 final|59.9263%|25.5162%|8.2050%|0.0000%|0.3081|80.7127%|794.13|
|cw_sweep10 initial|19.3082%|66.3125%|41.0918%|0.0002%|0.4119|80.7480%|797.55|
|cw_sweep10 final|60.0207%|25.4423%|8.2001%|0.0000%|0.3072|80.6299%|797.55|

ccw_sweep10: {'fibers': 489197, 'pieces': 489663, 'split_fibers': 454, 'events': 6606100, 'continuations': 466, 'no_observations_removed_by_splitting': True}; sampled epochs=10.000, peakGPU=7.376GiB.
Readout: broken threshold=0.5 (None=joint argmax), sign tolerance=0; feasible=True.
Final per-type: {'parallel': {'events': 5433896, 'retained': 489615, 'known': 452097, 'violations': 0, 'sign_violation_percent': 0.0}, 'perpendicular': {'events': 1172204, 'retained': 52416, 'known': 52362, 'violations': 0, 'sign_violation_percent': 0.0}}
Continuation links retained: 0/466; median/p95/p99(basevx): None
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.0%.
Fixed-charge audit: {'multiplier': 4, 'broken_threshold': 0.5, 'sign_tolerance': 0, 'original_violations': 4, 'remaining_violations': 0, 'newly_violating_events': 0, 'original_violations_disabled': 4, 'original_violations_corrected_while_retained': 0}
Quarter-turn exceedance: all buckets 0.2860%; chosen retained 6.8019%; probability mass 1.4254%.
Chosen retained |offset| median/p95/p99/max: [0.1047040530980028, 0.25907791621785686, 0.326678550888519, 1.8492747628136186]; minimum bucket separation -1.248581 turns.
Shared phase range [-10.55487700298401, 9.768573039520703]; actual retained anchor-q min/median/max [-0.8020687094367239, 51.22633365054551, 98.98914752575647]. Bucket IDs are relative to the unwrapped anchor.

cw_sweep10: {'fibers': 489197, 'pieces': 489663, 'split_fibers': 454, 'events': 6606100, 'continuations': 466, 'no_observations_removed_by_splitting': True}; sampled epochs=10.000, peakGPU=7.376GiB.
Readout: broken threshold=0.5 (None=joint argmax), sign tolerance=0; feasible=True.
Final per-type: {'parallel': {'events': 5433896, 'retained': 489417, 'known': 451870, 'violations': 0, 'sign_violation_percent': 0.0}, 'perpendicular': {'events': 1172204, 'retained': 52290, 'known': 52239, 'violations': 0, 'sign_violation_percent': 0.0}}
Continuation links retained: 0/466; median/p95/p99(basevx): None
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.0%.
Fixed-charge audit: {'multiplier': 4, 'broken_threshold': 0.5, 'sign_tolerance': 0, 'original_violations': 4, 'remaining_violations': 0, 'newly_violating_events': 0, 'original_violations_disabled': 4, 'original_violations_corrected_while_retained': 0}
Quarter-turn exceedance: all buckets 0.2850%; chosen retained 6.7474%; probability mass 1.4261%.
Chosen retained |offset| median/p95/p99/max: [0.10438183382067164, 0.2588989450565435, 0.32813760754929844, 2.872049817186318]; minimum bucket separation -1.157887 turns.
Shared phase range [-10.768194476813353, 9.69645055348355]; actual retained anchor-q min/median/max [-1.093700037960663, 51.34600325347032, 98.63826832558802]. Bucket IDs are relative to the unwrapped anchor.
