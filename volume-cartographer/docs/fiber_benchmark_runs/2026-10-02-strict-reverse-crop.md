# Strict reverse-consistency tolerance — raw40500,1024³ crop

Fixed full-prefix512/no-history, beam8, cache/model/GT cohort and hard-both BP objective. Only reverse-check radii change (normal/in-sheet, base voxels). Equal radii use the same metric with isotropic Euclidean tolerance; correspondence/progress rules remain unchanged. GT acceptance and reseeding unchanged. No short-piece filtering. Cached CPU work,128threads, RelWithDebInfo.

| Variant | Replay ↑ | BP constraint ↑ | Oracle accuracy ↑ | Non-oracle accuracy ↑ | Observed shift ↑ |
|---|---:|---:|---:|---:|---:|
| baseline20x40 | 100.0000% | -82.2666% | 84.6154% | 42.3077% | 70.8333% |
| ellipse10x20 | 100.0000% | -96.7907% | 80.7692% | 53.8462% | 79.1667% |
| ellipse5x10 | 100.0000% | -45.9732% | 76.9231% | 50.0000% | 83.3333% |
| isotropic20 | 100.0000% | -25.2365% | 80.7692% | 50.0000% | 83.3333% |
| isotropic10 | 100.0000% | -28.3286% | 80.7692% | 50.0000% | 79.1667% |
| isotropic5 | 100.0000% | -134.8829% | 84.6154% | 34.6154% | 58.3333% |

| Variant | Ordinary errors | Filtered stops | Reference retained % | Dense candidate retained % | Traces | Missing / ties / unbounded | Shifts / observed | Trace+replay+BP+independent s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline20x40 | 0 | 6 | 94.8385% | 42.0382% | 1661 | 2 / 0 / 0 | 7 / 24 | 231.95 |
| ellipse10x20 | 0 | 7 | 92.5096% | 37.8498% | 1905 | 2 / 0 / 0 | 5 / 24 | 515.58 |
| ellipse5x10 | 0 | 23 | 73.6400% | 31.6015% | 2227 | 2 / 0 / 0 | 4 / 24 | 377.11 |
| isotropic20 | 0 | 6 | 93.7463% | 38.5212% | 1879 | 2 / 0 / 0 | 4 / 24 | 366.87 |
| isotropic10 | 0 | 16 | 82.4413% | 34.2447% | 1988 | 2 / 0 / 0 | 5 / 24 | 367.55 |
| isotropic5 | 0 | 83 | 18.3887% | 11.4296% | 3378 | 2 / 1 / 1 | 10 / 24 | 376.92 |

Replay100/(ordinary errors+1) excludes filtered stops. Reference coverage is accepted reference arc; dense retained fraction uses that variant’s own candidates. Both may change nonmonotonically when filtering changes restart/coverage decisions. Oracle accuracy is GT-assisted. Independent accuracy uses26references; observed shift omits missing references and bridges gaps. Missing/tied/unbounded readouts and ladders are in JSON. Full26shift unavailable for missing evidence. Independent initial states must match that variant’s oracle-run initial solve.

Constraint score is −100×problematic/retained-fulfilled (closer to zero is better), not a removal percentage. Accuracy counts exact assignments among all26 references; shift is100×(1−offset changes/observed references). Non-oracle scores use the deterministic rank1 readout. Isotropic5 has an ambiguous reference8 (four equal best candidates in the evaluated range and an upper-unbounded best region); its accuracy/shift must not be interpreted as a uniquely determined ladder. Other new variants have no tied/unbounded readouts.

Interpretation: isotropic20 is the strongest overall candidate here: substantially better constraint and shift scores with near-baseline reference retention and the same six filtered stops. Half-ellipse10/20 has the highest non-oracle accuracy among the new variants but a worse constraint score than baseline. Isotropic5 over-filters severely. These are single-run crop results, not a rerun of the full91-piece reviewed benchmark; no production default has changed.

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261002_strict_reverse_crop_v1`. Approved command: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, control action `strict_reverse_crop`. One run per variant, prior baseline reused after replay validation.
