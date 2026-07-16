# Native 3D Trace2CP Product Candidate Scoring

Update native 3D Trace2CP search so each candidate step is selected by a
single product score instead of an additive direction/presence loss.

Requirements:

- For each candidate step direction, sample the model direction and presence at
  the candidate point.
- Match the current sampled direction against the candidate step direction.
- Match the candidate sampled direction against the candidate step direction,
  respecting the model's sign-ambiguous direction encoding by aligning signs
  before the dot product.
- Maximize:
  `dot(current_dir, candidate_step_dir) * dot(candidate_sampled_dir, candidate_step_dir) * candidate_presence`.
- Invalid candidate points must remain rejected.
- Remove the obsolete native 3D Trace2CP additive
  `--direction-weight`/`--presence-weight` candidate-selection knobs.
- The default trace step size is `4.0` selected-level voxels unless overridden
  by `--step-voxels`.
