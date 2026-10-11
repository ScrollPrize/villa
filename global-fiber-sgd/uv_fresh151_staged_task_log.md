# Fresh 151-winding solve

- User rejected warm expansion; that experiment is not a valid answer.
- Follow uv_fresh151_plan.md, independently reviewed. Fresh joint initialization
  from physical radius/phase/arc observations, no prior U/V state.
- Original sequence: 10,000 joint Adam steps, 4,000 sign refinement steps,
  then 10 categorical sweeps. Stage handoffs only reference this new experiment.
- Upstream normal/H/V/direction exclusions remain fixed. No reference evidence.
- Keep initial_sigma=3 in winding units. Extend geometric prior over 0–150;
  regenerate width regularizer too, not the cached100-vector.
- Existing100 initialization remains unchanged; earlier outputs preserved.
- 24 focused tests passed; independent code review approved fresh initialization,
  regenerated width priors, dynamic state slices and only within-run handoffs.
- Both joint stages launched via approved run_classic_fiber_evaluation.py configs
  run_uv_fresh151_ccw_joint.json / run_uv_fresh151_cw_joint.json.
- Combined regressions40passed. Both joint runs active on separate GPUs,
  observed11.2GiB/device and98% utilization. Initial state has zero offsets,
  1% broken probability, no rejected pieces, occupied indices4–146.
- Step1000: ~70s, retained length99.9968%, sampled sign errors18.38%/18.32%
  versus ~41% initially. Intermediate joint audits use100k fixed sampled events;
  final joint and charged/categorical audits use the full graph.
- Fresh joint stages completed exit0:657.33s CCW /661.30s CW, peak9.57GiB
  allocated. Full-graph retained length87.5555%/87.5279%; sign violations
  273,582/274,574 (6.0885%/6.1142%). Summary explicitly records fresh_radial,
  geometry_trainable=true, widths_trainable=true, winding_count151.
- Both sign-charge16 stages launched from ONLY these new joint solutions.
