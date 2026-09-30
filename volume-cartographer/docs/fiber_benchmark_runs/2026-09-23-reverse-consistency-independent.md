# Unassisted winding inference after reverse consistency

Same2048crop,40/80seed suppression,45degree gate; reverse20/40ellipse and40base backoff.
Initial BP is reference-free with no reference-driven pruning or conditioned
solve. Each reference is assigned independently against frozen initial states.
Existing global sign/offset gauge alignment uses GT for comparison; not a
held-out gauge. Existing defect-state inactive evidence handling remains unchanged.

|Metric|45° baseline|Plus reverse consistency|
|---|---:|---:|
|Correct top-ranked winding|26.92%(7/26)|**38.46%(10/26)**|
|No hard-sign violations|42.31%(11/26)|34.62%(9/26)|
|Correct and no hard-sign violations|15.38%(4/26)|7.69%(2/26)|

Correct references:0,0.5,1,1.5,2,4,4.5,5,5.5,6.
Improvement:3additional correct references,+11.54percentage points.
This remains far below the26/26GT-conditioned diagnostic; lower removal score
does not imply a globally correct unassisted winding ladder.

## Full ladder

|Reference|Inferred|Correct|Hard-sign errors|
|---:|---:|---|---:|
|0|0|yes|26|
|0.5|0.5|yes|15|
|1|1|yes|1|
|1.5|1.5|yes|13|
|2|2|yes|13|
|2.5|1.5|no|24|
|3|2|no|0|
|3.5|2.5|no|0|
|4|4|yes|0|
|4.5|4.5|yes|0|
|5|5|yes|15|
|5.5|5.5|yes|21|
|6|6|yes|27|
|6.5|5.5|no|3|
|7|6|no|0|
|7.5|6.5|no|35|
|8|6|no|8|
|8.5|6.5|no|0|
|9|7|no|28|
|9.5|6.5|no|0|
|10|7|no|6|
|10.5|7.5|no|7|
|11|8|no|0|
|11.5|9.5|no|0|
|12|10|no|23|
|12.5|9.5|no|6|

## Reproduction and validation

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_independent_reference_2048_reverse_20260923.json
```

Config differs from prior independent45°experiment only in input/output.
Independent review verified absence of conditioning/pruning flags.
No native change/build/retracing. Return0,116.0719s,RelWithDebInfo,single run.
Initial BP reached500-message cap,not convergence (residual532.9).
All6790piece validity,winding output and final-class fields match the initial
states of preceding reverse-consistency removal run exactly.

Output prefix:`/ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/bp-beam-new2048-coverage40x80-angle45-reverse-v2-independent`.
Log records top5ranked alternatives and costs per reference; all26assigned.
