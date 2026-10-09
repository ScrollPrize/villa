# Global U/V layout: first experiments

All earlier rejections fixed; baked H/V. Full-graph readout, no ground-truth layout labels.
Ordering errors use retained known-sign events (tolerance1e-4winding); finite penalty, not hard feasibility.
Rejection and arc retention use input pieces/arc after earlier stages. Distance errors are winding units.

| Run | Rejected pieces | Retained arc | Retained events | Sign errors | Mean distance error | Ambiguous (<90%) | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
|pilot_ccw_v2 initial|0.0000%|100.0000%|100.0000%|40.2544%|2.0387|100.0000%|14.47|
|pilot_ccw_v2 final|2.0390%|94.6229%|84.2771%|7.5081%|1.2595|95.8721%|14.47|
|pilot_expected initial|0.0000%|100.0000%|100.0000%|40.2544%|2.0387|100.0000%|14.52|
|pilot_expected final|2.8986%|92.9028%|80.0985%|7.9222%|0.7595|93.8531%|14.52|

pilot_ccw_v2: {'fibers': 10000, 'pieces': 10005, 'split_fibers': 5, 'events': 105178, 'continuations': 5, 'no_observations_removed_by_splitting': True}; sampled epochs=4.868, peakGPU=0.830GiB.
Final per-type: {'parallel': {'events': 86575, 'retained': 72026, 'known': 66445, 'violations': 1795, 'sign_violation_percent': 2.70148242907668}, 'perpendicular': {'events': 18603, 'retained': 16615, 'known': 16612, 'violations': 4441, 'sign_violation_percent': 26.73368649169275}}
Continuation links retained: 0/5; median/p95/p99(basevx): None
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.0%.
Quarter-turn exceedance: all buckets 0.8320%; chosen retained 14.9781%; probability mass 12.5269%.
Chosen retained |offset| median/p95/p99/max: [0.14738793671131134, 0.2966432273387909, 0.3695315718650818, 0.756312370300293]; minimum bucket separation 0.130259 turns.
Shared phase range [-4.319626318088926, 5.4248381060847635]; actual retained anchor-q min/median/max [2.5210765457768747, 51.87888704103535, 93.37124203155655]. Bucket IDs are relative to the unwrapped anchor.

pilot_expected: {'fibers': 10000, 'pieces': 10005, 'split_fibers': 5, 'events': 105178, 'continuations': 5, 'no_observations_removed_by_splitting': True}; sampled epochs=4.868, peakGPU=0.790GiB.
Final per-type: {'parallel': {'events': 86575, 'retained': 68199, 'known': 62873, 'violations': 2016, 'sign_violation_percent': 3.206463823898971}, 'perpendicular': {'events': 18603, 'retained': 16047, 'known': 16044, 'violations': 4236, 'sign_violation_percent': 26.402393418100225}}
Continuation links retained: 0/5; median/p95/p99(basevx): None
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.0%.
Quarter-turn exceedance: all buckets 1.0365%; chosen retained 17.0767%; probability mass 16.2064%.
Chosen retained |offset| median/p95/p99/max: [0.15587227046489716, 0.3014000058174132, 0.38104433476924976, 0.7679190039634705]; minimum bucket separation 0.179015 turns.
Shared phase range [-4.191880763156987, 5.2995632266404495]; actual retained anchor-q min/median/max [3.5250060520445894, 51.82235777419475, 92.9853821695866]. Bucket IDs are relative to the unwrapped anchor.
