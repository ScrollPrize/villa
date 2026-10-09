# Global U/V layout: first experiments

All earlier rejections fixed; baked H/V. Full-graph readout, no ground-truth layout labels.
Ordering errors use retained known-sign events; each run records its tolerance and broken readout below. Finite penalty is not a hard feasibility guarantee.
Rejection and arc retention use input pieces/arc after earlier stages. Distance errors are winding units.

| Run | Rejected pieces | Retained arc | Retained events | Sign errors | Mean distance error | Ambiguous (<90%) | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
|pilot_t0.25 initial|1.5692%|95.6542%|86.7254%|7.6367%|1.3353|95.8721%|19.04|
|pilot_t0.25 final|2.4588%|93.6321%|83.5213%|3.9873%|1.1531|73.4433%|19.04|
|pilot_t1 initial|1.5692%|95.6542%|86.7254%|7.6367%|1.3353|95.8721%|19.12|
|pilot_t1 final|6.4168%|87.5463%|75.6422%|3.8497%|1.2435|91.7341%|19.12|
|pilot_damped initial|1.5692%|95.6542%|86.7254%|7.6367%|1.3353|95.8721%|120.49|
|pilot_damped final|32.0040%|52.7570%|11.3008%|0.0000%|0.4871|55.8921%|120.49|
|pilot_repair initial|18.8906%|69.4125%|44.3981%|0.5106%|0.4856|43.9180%|120.13|
|pilot_repair final|16.6517%|71.9722%|47.4881%|0.0000%|0.3515|44.6577%|120.13|

pilot_t0.25: {'fibers': 10000, 'pieces': 10005, 'split_fibers': 5, 'events': 105178, 'continuations': 5, 'no_observations_removed_by_splitting': True}; sampled epochs=30.000, peakGPU=1.111GiB.
Readout: broken threshold=0.5 (None=joint argmax), sign tolerance=0; feasible=False.
Final per-type: {'parallel': {'events': 86575, 'retained': 71665, 'known': 66283, 'violations': 618, 'sign_violation_percent': 0.9323657649774452}, 'perpendicular': {'events': 18603, 'retained': 16181, 'known': 16178, 'violations': 2670, 'sign_violation_percent': 16.503894177277786}}
Continuation links retained: 0/5; median/p95/p99(basevx): None
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.040987806127677016%.
Fixed-charge audit: {'multiplier': 4, 'broken_threshold': 0.5, 'sign_tolerance': 0, 'original_violations': 6525, 'remaining_violations': 3288, 'newly_violating_events': 910, 'original_violations_disabled': 1033, 'original_violations_corrected_while_retained': 3114, 'remaining_violation_magnitude_quantiles': [0.002539442942310899, 2.050926244446046, 7.870694333559619, 20.204718684221795], 'remaining_max_endpoint_broken_probability_quantiles': [0.0002750955005290938, 0.021215332111317643, 0.026771239755814302, 0.41527270145914985], 'remaining_unique_endpoint_pieces': 3588, 'remaining_unique_source_fibers': 3588, 'remaining_endpoint_broken_probability_quantile_levels': [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], 'remaining_endpoint_broken_probability_quantiles': [2.2367811318313097e-05, 0.010489137066052966, 0.012499770429275234, 0.018656953773610874, 0.022228942439137117, 0.026524420543107875, 0.41527270145914985], 'remaining_endpoint_broken_probability_mean': 0.01829546973297803, 'remaining_endpoint_broken_probability_bin_edges': [0.0, 0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 1.000001], 'remaining_endpoint_broken_probability_bin_counts': [138, 3447, 1, 0, 1, 1, 0], 'remaining_max_endpoint_broken_probability_mean': 0.02209077772493167, 'remaining_min_endpoint_broken_probability_quantiles': [2.2367811318313097e-05, 0.013856002328489247, 0.022324973953933347, 0.02710454310259571], 'remaining_pairs_both_below_probability': {'0.01': 31, '0.05': 3274, '0.1': 3275, '0.25': 3275, '0.4': 3281}, 'all_retained_piece_broken_probability_quantiles': [2.2879861104939013e-06, 0.010182495889921726, 0.02533361027243536, 0.41527270145914985], 'all_retained_piece_broken_probability_mean': 0.010109985853659867}
Quarter-turn exceedance: all buckets 0.8320%; chosen retained 9.4989%; probability mass 9.2752%.
Chosen retained |offset| median/p95/p99/max: [0.13880571364890784, 0.26827413447899745, 0.3282728705584538, 0.7989251781068742]; minimum bucket separation 0.130259 turns.
Shared phase range [-4.319626318088926, 5.4248381060847635]; actual retained anchor-q min/median/max [0.025005965110792904, 51.56313658582175, 94.76447735566084]. Bucket IDs are relative to the unwrapped anchor.

pilot_t1: {'fibers': 10000, 'pieces': 10005, 'split_fibers': 5, 'events': 105178, 'continuations': 5, 'no_observations_removed_by_splitting': True}; sampled epochs=30.000, peakGPU=1.111GiB.
Readout: broken threshold=0.5 (None=joint argmax), sign tolerance=0; feasible=False.
Final per-type: {'parallel': {'events': 86575, 'retained': 65898, 'known': 60815, 'violations': 490, 'sign_violation_percent': 0.8057222724656746}, 'perpendicular': {'events': 18603, 'retained': 13661, 'known': 13658, 'violations': 2377, 'sign_violation_percent': 17.403719431834823}}
Continuation links retained: 0/5; median/p95/p99(basevx): None
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.03204101249599487%.
Fixed-charge audit: {'multiplier': 4, 'broken_threshold': 0.5, 'sign_tolerance': 0, 'original_violations': 6525, 'remaining_violations': 2867, 'newly_violating_events': 802, 'original_violations_disabled': 1922, 'original_violations_corrected_while_retained': 2538, 'remaining_violation_magnitude_quantiles': [0.00038028161188208287, 1.9646258542082649, 7.947646861282359, 21.23250077560469], 'remaining_max_endpoint_broken_probability_quantiles': [0.0024027671333865107, 0.028545056480420525, 0.24094660407580662, 0.46530248903952154], 'remaining_unique_endpoint_pieces': 3229, 'remaining_unique_source_fibers': 3229, 'remaining_endpoint_broken_probability_quantile_levels': [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], 'remaining_endpoint_broken_probability_quantiles': [0.00142313656583231, 0.017964377379684728, 0.022591687366505644, 0.025646529526470143, 0.028855161877318434, 0.08627374392502464, 0.46530248903952154], 'remaining_endpoint_broken_probability_mean': 0.0351919714069052, 'remaining_endpoint_broken_probability_bin_edges': [0.0, 0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 1.000001], 'remaining_endpoint_broken_probability_bin_counts': [16, 2971, 93, 96, 44, 9, 0], 'remaining_max_endpoint_broken_probability_mean': 0.05417461927012041, 'remaining_min_endpoint_broken_probability_quantiles': [0.00142313656583231, 0.02290044471399892, 0.029426318892195984, 0.18595440382348374], 'remaining_pairs_both_below_probability': {'0.01': 4, '0.05': 2341, '0.1': 2523, '0.25': 2726, '0.4': 2841}, 'all_retained_piece_broken_probability_quantiles': [1.7596651743759515e-05, 0.02165482647310068, 0.04359031647634363, 0.49044809189441674], 'all_retained_piece_broken_probability_mean': 0.023049870775877716}
Quarter-turn exceedance: all buckets 0.8320%; chosen retained 9.2065%; probability mass 8.0298%.
Chosen retained |offset| median/p95/p99/max: [0.13817398324841634, 0.2658098358090501, 0.32859600343704193, 0.7989251781068742]; minimum bucket separation 0.130259 turns.
Shared phase range [-4.319626318088926, 5.4248381060847635]; actual retained anchor-q min/median/max [0.07373896109528812, 51.35215251846339, 94.7287672349708]. Bucket IDs are relative to the unwrapped anchor.

pilot_damped: {'fibers': 10000, 'pieces': 10005, 'split_fibers': 5, 'events': 105178, 'continuations': 5, 'no_observations_removed_by_splitting': True}; sampled epochs=100.000, peakGPU=1.111GiB.
Readout: broken threshold=0.5 (None=joint argmax), sign tolerance=0; feasible=True.
Final per-type: {'parallel': {'events': 86575, 'retained': 8537, 'known': 7641, 'violations': 0, 'sign_violation_percent': 0.0}, 'perpendicular': {'events': 18603, 'retained': 3349, 'known': 3348, 'violations': 0, 'sign_violation_percent': 0.0}}
Continuation links retained: 0/5; median/p95/p99(basevx): None
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.0%.
Fixed-charge audit: {'multiplier': 4, 'broken_threshold': 0.5, 'sign_tolerance': 0, 'original_violations': 6525, 'remaining_violations': 0, 'newly_violating_events': 0, 'original_violations_disabled': 6129, 'original_violations_corrected_while_retained': 396}
Quarter-turn exceedance: all buckets 0.8320%; chosen retained 7.2027%; probability mass 7.2726%.
Chosen retained |offset| median/p95/p99/max: [0.1309443278517574, 0.2571101552490145, 0.3099445216207062, 0.5197976181190461]; minimum bucket separation 0.130259 turns.
Shared phase range [-4.319626318088926, 5.4248381060847635]; actual retained anchor-q min/median/max [0.642565045553335, 52.09379774321129, 94.79587410760419]. Bucket IDs are relative to the unwrapped anchor.

pilot_repair: {'fibers': 10000, 'pieces': 10005, 'split_fibers': 5, 'events': 105178, 'continuations': 5, 'no_observations_removed_by_splitting': True}; sampled epochs=100.000, peakGPU=1.111GiB.
Readout: broken threshold=0.5 (None=joint argmax), sign tolerance=0; feasible=True.
Final per-type: {'parallel': {'events': 86575, 'retained': 41782, 'known': 39383, 'violations': 0, 'sign_violation_percent': 0.0}, 'perpendicular': {'events': 18603, 'retained': 8165, 'known': 8162, 'violations': 0, 'sign_violation_percent': 0.0}}
Continuation links retained: 0/5; median/p95/p99(basevx): None
Numerically zero probabilities: 0.000000%; boundary hypotheses: 0.0%.
Fixed-charge audit: {'multiplier': 4, 'broken_threshold': 0.5, 'sign_tolerance': 0, 'original_violations': 227, 'remaining_violations': 0, 'newly_violating_events': 0, 'original_violations_disabled': 90, 'original_violations_corrected_while_retained': 137}
Quarter-turn exceedance: all buckets 0.1388%; chosen retained 4.0053%; probability mass 4.0852%.
Chosen retained |offset| median/p95/p99/max: [0.13019135342910887, 0.2478900671717711, 0.27006363767571767, 0.5564521664567292]; minimum bucket separation 0.238298 turns.
Shared phase range [-6.976908358377062, 6.837892146466652]; actual retained anchor-q min/median/max [2.9271589646731324, 51.749598494802946, 93.61477606849418]. Bucket IDs are relative to the unwrapped anchor.
