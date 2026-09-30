# Completed cost sweep: ladder shift diagnostic

All 32 trials completed. Derived directly from saved rank-1 independent-reference assignments of the unconditioned BP solves; no re-solving.

For reference-sorted assignments, let e_i = predicted_i - reference_i. K counts adjacent changes in e_i. Ignore a global offset; each change requires a suffix shift of arbitrary magnitude. Score = 100*(1-K/N), matching the requested 25/26 for one inserted winding. This is not unit-winding edit distance and not adjacent-gap accuracy (which has denominator N-1).

Removal = negative problematic/fulfilled constraint percentage, closer to zero is better. Initial is unconditioned independent winding accuracy; conditioned is the separate reference-conditioned benchmark. Neither is interchangeable with the shift score.

| Mode | Parallel | D | C | Removal % | Initial % | Conditioned % | Offset changes | Shift score % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| joint | off | 1 | 0.1 | -1153.95 | 3.85 | 19.23 | 25/25 | 3.85 |
| joint | off | 1 | 1 | -2193.93 | 0.00 | 19.23 | 23/25 | 11.54 |
| joint | off | 1 | 10 | -1604.88 | 11.54 | 34.62 | 24/25 | 7.69 |
| joint | off | 1 | hard | -1736.17 | 0.00 | 34.62 | 25/25 | 3.85 |
| joint | off | 10 | 0.1 | -971.18 | 3.85 | 23.08 | 24/25 | 7.69 |
| joint | off | 10 | 1 | -48.86 | 3.85 | 30.77 | 25/25 | 3.85 |
| joint | off | 10 | 10 | -20.66 | 7.69 | 42.31 | 22/25 | 15.38 |
| joint | off | 10 | hard | -21.83 | 7.69 | 42.31 | 23/25 | 11.54 |
| joint | off | 100 | 0.1 | -999.85 | 0.00 | 38.46 | 25/25 | 3.85 |
| joint | off | 100 | 1 | -38.38 | 3.85 | 26.92 | 25/25 | 3.85 |
| joint | off | 100 | 10 | -24.94 | 11.54 | 42.31 | 24/25 | 7.69 |
| joint | off | 100 | hard | -23.81 | 11.54 | 26.92 | 24/25 | 7.69 |
| joint | on | 1 | 0.1 | -1079.76 | 7.69 | 46.15 | 25/25 | 3.85 |
| joint | on | 1 | 1 | -2215.99 | 11.54 | 26.92 | 24/25 | 7.69 |
| joint | on | 1 | 10 | -1627.73 | 11.54 | 34.62 | 25/25 | 3.85 |
| joint | on | 1 | hard | -1876.99 | 15.38 | 15.38 | 24/25 | 7.69 |
| joint | on | 10 | 0.1 | -1056.98 | 15.38 | 19.23 | 23/25 | 11.54 |
| joint | on | 10 | 1 | -41.88 | 11.54 | 34.62 | 25/25 | 3.85 |
| joint | on | 10 | 10 | -22.78 | 15.38 | 30.77 | 22/25 | 15.38 |
| joint | on | 10 | hard | -20.42 | 15.38 | 15.38 | 22/25 | 15.38 |
| joint | on | 100 | 0.1 | -1106.98 | 11.54 | 30.77 | 25/25 | 3.85 |
| joint | on | 100 | 1 | -48.79 | 3.85 | 46.15 | 25/25 | 3.85 |
| joint | on | 100 | 10 | -24.09 | 7.69 | 50.00 | 23/25 | 11.54 |
| joint | on | 100 | hard | -25.91 | 7.69 | 46.15 | 24/25 | 7.69 |
| fixed | off | 100 | 0.1 | -525.90 | 11.54 | 30.77 | 23/25 | 11.54 |
| fixed | off | 100 | 1 | -69.29 | 11.54 | 30.77 | 22/25 | 15.38 |
| fixed | off | 100 | 10 | -54.81 | 15.38 | 19.23 | 20/25 | 23.08 |
| fixed | off | 100 | hard | -74.79 | 11.54 | 34.62 | 22/25 | 15.38 |
| fixed | on | 100 | 0.1 | -530.34 | 11.54 | 34.62 | 25/25 | 3.85 |
| fixed | on | 100 | 1 | -60.67 | 15.38 | 26.92 | 21/25 | 19.23 |
| fixed | on | 100 | 10 | -64.80 | 15.38 | 50.00 | 22/25 | 15.38 |
| fixed | on | 100 | hard | -90.50 | 15.38 | 57.69 | 20/25 | 23.08 |

These ladders show widespread local disagreement rather than one persistent offset. A global offset or a single inserted winding cannot explain the low accuracy. Causes are not isolated by this sweep: objective/readout mismatch, weakly determined or contradictory constraints, and approximate BP decoding remain candidates. Many runs hit the message limit; this is not proof of optimal solutions.

