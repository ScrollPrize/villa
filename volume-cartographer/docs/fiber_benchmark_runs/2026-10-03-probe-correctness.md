# Two-sided probe correctness audit

This geometric subset audit does **not** answer the requested reference-based
constraint-removal comparison. Use [the established removal benchmark](2026-10-03-probe-removal.md)
for that comparison. Retained here only as a separately labelled diagnostic.

Same cached raw40500 iso20 1024 crop and weight0.05 legacy-fine256 winner as
[the probe experiment](2026-10-02-iso20-orthogonal-probes.md). No inference,
tracing, BP rerun, or production changes. All three variants retain61.54%
independent exact accuracy and87.50% observed shift score.

## Ground-truth correspondence and limits

Reuse the shared ordered OBJ reader. Source curves come from winner-repeat_initial.obj;
piece/source identities and arc intervals come from consistency.csv. Reference
curves are native reference_hs_N.obj exports; N/2 is the benchmark reference
winding (native reference comparison uses the same convention).

Sample each piece at equal arc-weight midpoints spaced at most4 base voxels.
Require >=90% samples within radius and <=25-degree sign-invariant tangent
agreement with one reference. Any sample matching another label rejects the
piece. Each factor's actual A/B arc contact must independently match that same
unique label. Distances are exact point-to-segment distances after conservative
KD-tree segment-midpoint broad phase. This spherical 4/8/16vx audit is NOT the
acceptance ellipse and does not replace a benchmark metric.

Normal polarity is calibrated once per component using baseline matched signs,
then frozen for both probe variants. All support is component0, polarity+1;
all matched signs agree. Tied polarity is unknown, not optimized per variant.
This calibration and geometric matching mean absolute errors are conditional
on correspondence accuracy. Unmatched factors are unknown, not correct.
Counts below are repeated stored measurements, not independent fiber pairs.

## Wrong hard ordering signs

| Match radius (base vx) | Matched pieces /2799 | Orthogonal assessed | Wrong, no probes | Wrong, median64 | Wrong, consensus32 | Parallel assessed | Parallel wrong (each) |
|---:|---:|---:|---:|---:|---:|---:|---:|
|4|25|8|0 (0%)|0 (0%)|0 (0%)|0|N/A|
|8|96|198|0 (0%)|0 (0%)|0 (0%)|88|0 (0%)|
|16|129|468|0 (0%)|0 (0%)|0 (0%)|133|0 (0%)|

Same factor identities and eligibility across variants. Zero corrected and
zero newly damaged signs on this subset. At16vx,468/62517=0.7486% of prepared
hard orthogonal factors are assessable;62049 remain unknown. No global
wrong-constraint percentage can be inferred from this selected subset.
Parallel targets are unchanged and asserted identical across variants.

## Orthogonal magnitude (diagnostic only)

Mean absolute error of uncalibrated absolute measured winding difference
against absolute reference winding difference; not a binary wrong/right count.

| Radius | No probes | Median64 | Consensus32 |
|---:|---:|---:|---:|
|4|0.279597|0.269306|0.279597|
|8|0.541057|0.523258|0.542682|
|16|0.423880|0.411103|0.422185|

Median64 improves raw magnitude MAE about3.0% at16vx. It changes228/468 assessed
targets; consensus32 changes91/468. Neither changes any assessed ordering sign.
The current objective gives perpendicular magnitude zero weight, so improving
its numerical estimate without changing signs does not directly improve that
objective. Do not treat these raw magnitudes as calibrated integer winding
estimates. The no-probe winner remains preferable on demonstrated scores.

## Reproduction and validation

Set scripts/training_failure_mining_control.json action to
`iso20_probe_correctness`, then run the approved command:

```sh
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Script: scripts/iso20_probe_correctness.py. Detailed data in the adjacent JSON.
Source artifacts:
`/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_orthogonal_probes_v1`.

```sh
/home/hendrik/.venv_las/bin/python -m pytest -q scripts/test_iso20_probe_correctness.py scripts/test_iso20_orthogonal_probes.py
```

Seven tests passed: exact sampling, reversed tangents, ambiguity including a
short competing segment, frozen polarity, unknown calibration, contact checks,
and existing probe matrix tests. Native code unchanged; no native rebuild.
Independent plan review added strict local contact checks and rejection of
even short competing-label matches. No new oracle-removal run or exhaustive
GT assignment of the remaining graph was performed.
