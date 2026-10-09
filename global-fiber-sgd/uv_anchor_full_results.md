# Global U/V layout: first experiments

All earlier rejections fixed; baked H/V. Full-graph readout, no ground-truth layout labels.
Ordering errors use retained known-sign events (tolerance1e-4winding); finite penalty, not hard feasibility.
Rejection and arc retention use input pieces/arc after earlier stages. Distance errors are winding units.

| Run | Rejected pieces | Retained arc | Retained events | Sign errors | Mean distance error | Ambiguous (<90%) | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
|ccw initial|0.0000%|100.0000%|100.0000%|41.7028%|1.8172|100.0000%|1022.35|
|ccw final|3.3374%|91.4355%|82.6505%|6.4448%|0.8349|54.1901%|1022.35|
|cw initial|0.0000%|100.0000%|100.0000%|41.5456%|1.8236|100.0000%|1023.32|
|cw final|3.3686%|91.3679%|82.5233%|6.4855%|0.8362|54.1724%|1023.32|

ccw: {'fibers': 489197, 'pieces': 489663, 'split_fibers': 454, 'events': 6606100, 'continuations': 466, 'no_observations_removed_by_splitting': True}; sampled epochs=6.200, peakGPU=5.143GiB.
Final per-type: {'parallel': {'events': 5433896, 'retained': 4501995, 'known': 3994198, 'violations': 139765, 'sign_violation_percent': 3.499200590456457}, 'perpendicular': {'events': 1172204, 'retained': 957980, 'known': 957453, 'violations': 179358, 'sign_violation_percent': 18.732825527728252}}
Continuation links retained: 40/466; median/p95/p99(basevx): [20045.474466815267, 159449.50893433348, 322523.3479655242]
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.0%.
Quarter-turn exceedance: all buckets 0.3279%; chosen retained 11.9900%; probability mass 11.1055%.
Chosen retained |offset| median/p95/p99/max: [0.139018252491951, 0.28510141372680664, 0.36389742493629446, 1.0617282390594482]; minimum bucket separation -0.420550 turns.
Shared phase range [-9.585691173821413, 9.583177549557252]; actual retained anchor-q min/median/max [-1.5823429524119508, 49.85541346833544, 98.84430115081848]. Bucket IDs are relative to the unwrapped anchor.

cw: {'fibers': 489197, 'pieces': 489663, 'split_fibers': 454, 'events': 6606100, 'continuations': 466, 'no_observations_removed_by_splitting': True}; sampled epochs=6.200, peakGPU=5.143GiB.
Final per-type: {'parallel': {'events': 5433896, 'retained': 4495034, 'known': 3987851, 'violations': 141154, 'sign_violation_percent': 3.5396006520805314}, 'perpendicular': {'events': 1172204, 'retained': 956541, 'known': 956012, 'violations': 179481, 'sign_violation_percent': 18.773927523922293}}
Continuation links retained: 35/466; median/p95/p99(basevx): [14858.09095264143, 208752.32351193356, 337271.40095713997]
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.0%.
Quarter-turn exceedance: all buckets 0.3299%; chosen retained 12.0887%; probability mass 11.1936%.
Chosen retained |offset| median/p95/p99/max: [0.1394694447517395, 0.2851664423942565, 0.3642314291000368, 1.0605616569519043]; minimum bucket separation -0.628276 turns.
Shared phase range [-9.764014522284544, 8.91368772296472]; actual retained anchor-q min/median/max [-2.516795890503231, 49.89130010061865, 98.82470495618306]. Bucket IDs are relative to the unwrapped anchor.
