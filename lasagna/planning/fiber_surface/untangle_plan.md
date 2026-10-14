# Explicit signed-area untangling

User approved direct untangling followed by feasibility-preserving continuation.
Reuse both 3x-smoothness surfaces, selected H fibers and saved attachments.

1. Reuse CurveBank normal sampling and inward sign solution. Interpolate aligned
   normals onto the UV grid (nearest valid observations, then local smoothing).
   Choose ONE global orientation sign to match the UV chart; never flip each
   cell to agree with the folded mesh. Save field/provenance and coverage stats.
2. Optimize signed bilinear corner Jacobians directly, at fixed observation UV.
   Active connected defect patches have halos8 then16/32 if needed. Healthy
   surrounding vertices can move within the halo; outside remains exactly fixed.
   Minimize signed-area hinge violations first, with weak positional/fiber
   guidance. Allow intermediate inversions; evaluate final feasibility/fit.
   Pilot a few worst real patches, then both full regional meshes if viable.
3. Maintain positivity by backtracking the existing optimizer's proposed joint
   mesh/UV step. Protect all initially feasible corners and newly repaired ones.
   If entire mesh cannot be made feasible, clearly report partial protection and
   unresolved violations. Roll back infeasible proposals after bounded retries.
   No global collision guarantee. Fresh Adam state, original UV reference kept.
4. Test synthetic inversion recovery, fixed boundary, aligned-normal interpolation,
   invalid source normals, backtracking/rejection, immutable source XYZ/UV during
   untangling. Compare signed-corner violation counts and previous fold metrics,
   mean/p95/max/normal errors, moved distances and rejected continuation steps.
5. Use approved runner, non-exclusive8GiB grants on GPUs0/1 (allowed0–5).
   Save before/untangled/continued tifxyz, attachments, histories and field.

## Spec update

Opt-in explicit geometric feasibility stage. Uses existing signed fiber normals
as fixed orientation evidence, not folded surface normals. Normal-only data loss
and joint UV updates resume after untangling. No default changes or data rejection.

## Documentation, changelog and tests

Update task/spec/status/log, add untangle_results.md with all trials, exact configs
and commands, limitations (normal evidence, local validity, any failed patches).
Run existing regression suite. Independent plan/code review required.
