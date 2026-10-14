# Patch repair experiments — 2026-10-14

## Outcome

Larger patches alone made little difference. Adding boundary-slope preservation
and partial steps improved immediate fold counts from225/233 to183/184
(original/deformed), but continued optimization can undo part of this gain.
No variant eliminates folds. No defaults promoted; keep these experimental.

- Small-first (halos2/4/8):209/206 immediately,195/200 after continuation.
- Large-first (16/8/4/2, largest components first):207/205, then189/200.
- Large-first + seam slope penalty10 + step fractions1/.5/.25:183/184,
  then200/195. Mean post-continuation fit6.919/6.775base vx.
- Same repair + guard10000 instead of100:208/195 after continuation; stronger
  guard did not reliably preserve the repaired folds and is not recommended.

The immediate best repair reduces folds18.7%/21.0%, with mean XYZ error increasing
7.931→8.254 and7.995→8.519base vx. Repair-only output is worth inspection but
is not a finished untangled reconstruction. The guard constrains reference corner
orientation softly; it is not an injectivity barrier or global collision check.

## Inputs and matched comparisons

Both saved smooth1.5/bend5 regional surfaces,849 identical retained H pieces,
grid100base voxels. Original has397184vertices/43478attachments; deformed has
416016/43496. Input roots:
- lasagna/data/smoothing_original_b5_s1.5_v1
- lasagna/data/smoothing_deformed_b5_s1.5_v1

All UV positions frozen during repair; source XYZ never changes. Restore saved
final UV for continuation, but retain original UV as the displacement reference.
Both control and repaired continuation use existing Lasagna mesh pyramid and
joint optimizer,600steps,LR1,constant dataweight20,smooth1.5,bend5,step.5.
Each gets its own healthy-reference guard and a fresh Adam state. These are not
exact optimizer-checkpoint resumes.

Each variant has a fresh no-repair control. Repeated GPU controls vary slightly
(e.g. original198/203/200 folds at guard100); do not interpret a handful of
folds as a proven improvement. No deterministic CUDA execution was forced.
The tangent/strong-guard runs include an additional eligibility correction
excluding opposing diagonal triangles from guard references; code hashes are
saved per run. Immediate CPU repair counts are directly comparable.

## Reconstruction and acceptance

Defects seed connected regions on the UV grid. Grow halos, freeze all patch
boundary vertices (also outer mesh boundary), solve sparse least-squares
harmonic interior with arc-weighted bilinear XYZ guides at fixed UV. Guide
weights0/.1/1 are tried. Stronger seam variant penalizes changes of the frozen
boundary's Laplacian, preserving incoming slope, weight10. It tries moving
1,.5,.25 toward the fitted interior and selects the best acceptable candidate.

Accept only finite candidates with fewer total defect entries, no new defective
cells or neighbor edges, local arc-weighted mean error <=1.25*before+5vx and
global mean <=original+1vx. Full boundary remains bit-identical. Candidate
audits cover all changed cells and neighboring seams using a bounded local view.

Small-first accepted44/54 patches out of466/503 patch visits.
Tangent/damped variant accepted112/132. Most rejected candidates introduce
a new seam defect; fit guards also reject candidates. Counts are in metrics JSON.

Metrics retain the earlier opposing-two-triangle count. Additional corner
Jacobian tests catch bilinear concavity/collapse missed by that count (area
threshold step²*1e-4). Neighbor reversals count adjacent mean normals with
negative dot; these are diagnostics, not proof that each is a physical error.
All saved surfaces are finite; old absolute degenerate count is zero.

## All measured comparisons

| Input | Variant | Stage | Folded quads | Corner-defective cells | Neighbor reversals | Mean XYZ vx | P95 XYZ vx | Normal RMS vx |
|---|---|---|---:|---:|---:|---:|---:|---:|
| original | repair | before | 225 | 305 | 964 | 7.931 | 21.839 | 2.313 |
| original | repair | repaired | 209 | 277 | 856 | 7.970 | 22.065 | 2.567 |
| original | repair | control | 198 | 269 | 807 | 6.908 | 19.617 | 0.888 |
| original | repair | continued | 195 | 246 | 762 | 6.889 | 19.576 | 0.902 |
| deformed | repair | before | 233 | 311 | 1059 | 7.995 | 21.575 | 2.687 |
| deformed | repair | repaired | 206 | 266 | 929 | 8.057 | 21.863 | 2.917 |
| deformed | repair | control | 216 | 275 | 929 | 6.755 | 19.367 | 0.927 |
| deformed | repair | continued | 200 | 269 | 863 | 6.789 | 19.535 | 0.964 |
| original | repair_large | before | 225 | 305 | 964 | 7.931 | 21.839 | 2.313 |
| original | repair_large | repaired | 207 | 275 | 857 | 7.899 | 22.039 | 2.701 |
| original | repair_large | control | 203 | 263 | 815 | 6.880 | 19.643 | 0.877 |
| original | repair_large | continued | 189 | 242 | 762 | 6.927 | 19.822 | 0.879 |
| deformed | repair_large | before | 233 | 311 | 1059 | 7.995 | 21.575 | 2.687 |
| deformed | repair_large | repaired | 205 | 265 | 929 | 8.169 | 22.298 | 3.558 |
| deformed | repair_large | control | 214 | 282 | 933 | 6.758 | 19.481 | 0.888 |
| deformed | repair_large | continued | 200 | 262 | 884 | 6.766 | 19.498 | 0.979 |
| original | repair_tangent | before | 225 | 305 | 964 | 7.931 | 21.839 | 2.313 |
| original | repair_tangent | repaired | 183 | 234 | 758 | 8.254 | 23.647 | 3.471 |
| original | repair_tangent | control | 200 | 258 | 794 | 6.908 | 19.726 | 0.875 |
| original | repair_tangent | continued | 200 | 255 | 767 | 6.919 | 19.749 | 0.887 |
| deformed | repair_tangent | before | 233 | 311 | 1059 | 7.995 | 21.575 | 2.687 |
| deformed | repair_tangent | repaired | 184 | 230 | 839 | 8.519 | 24.027 | 4.402 |
| deformed | repair_tangent | control | 215 | 281 | 948 | 6.757 | 19.403 | 0.922 |
| deformed | repair_tangent | continued | 195 | 253 | 875 | 6.775 | 19.530 | 0.926 |
| original | repair_guard | before | 225 | 305 | 964 | 7.931 | 21.839 | 2.313 |
| original | repair_guard | repaired | 183 | 234 | 758 | 8.254 | 23.647 | 3.471 |
| original | repair_guard | control | 228 | 298 | 756 | 7.006 | 20.233 | 1.010 |
| original | repair_guard | continued | 208 | 283 | 725 | 6.919 | 19.743 | 1.054 |
| deformed | repair_guard | before | 233 | 311 | 1059 | 7.995 | 21.575 | 2.687 |
| deformed | repair_guard | repaired | 184 | 230 | 839 | 8.519 | 24.027 | 4.402 |
| deformed | repair_guard | control | 231 | 303 | 857 | 6.737 | 19.432 | 0.899 |
| deformed | repair_guard | continued | 195 | 271 | 796 | 6.756 | 19.457 | 1.027 |

## Runtime and outputs

Single measured trials, no throughput claim: small-first full experiment33.5/35.5s;
large-first36.4/35.4s; tangent51.2/55.1s; strongguard51.9/54.7s.
Totals include input IO, repair, both continuations and saving; not manager startup.
Tangent reconstruction including input IO takes33.4/35.6s; GPU continuation
approximately8–10s each. Shared8GiB GPU grants on0/1,8CPUthreads per run, no
exclusive allocation. Python/NumPy/SciPy sparse CPU repair, existing PyTorch
CUDA optimizer, no new compiled build or dependency installation.

Each output contains before.tifxyz,repaired.tifxyz,control.tifxyz,continued.tifxyz,
LZW compressed; summary/config/progress and control/continued attachment NPZs.
Root pattern `lasagna/data/repair_{original,deformed}_v2`,
`repair_large_{original,deformed}_v1`, `repair_tangent_{original,deformed}_v1`,
`repair_guard_{original,deformed}_v1`.

Recommended repair-only inspection:
- /home/hendrik/villa4/lasagna/data/repair_tangent_original_v1/repaired.tifxyz
- /home/hendrik/villa4/lasagna/data/repair_tangent_deformed_v1/repaired.tifxyz

## Commands and verification

Use the approved runner; config pins inputs, grant and output. For another run,
use a fresh output_root (existing outputs are rejected).

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/repair_tangent_original.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/repair_tangent_deformed.json
```

Other exact configs are repair_original/deformed.json,
repair_large_original/deformed.json,repair_guard_original/deformed.json.

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --tb=short --noconftest -o pythonpath=/home/hendrik/villa4/vesuvius/src lasagna/tests/test_fiber_surface_repair.py lasagna/tests/test_sliding_fiber_surface.py lasagna/tests/test_fiber_surface.py lasagna/tests/test_opt_loss_step.py lasagna/tests/test_optimizer_stage_weights.py vesuvius/tests/test_fiber_uv_surface.py vesuvius/tests/test_fiber_lasagna_surface.py -k 'not atlas_snap_flow_speculative'
```

59passed,1previously documented unrelated deselected. Synthetic tests cover
fold recovery, exact frozen boundary, immutable UV, failed-fit rejection,
unrepairable boundary, saved UV restoration, guard gradients/near-collapse,
and tangent/blended mode. Actual outputs audited for unchanged outer boundary
and zero introduced defects from repair itself. Continuation DOES introduce
new local defects despite net reductions; soft guard is insufficient.

Initial repair_original_v1/repair_deformed_v1 were interrupted incomplete pilots,
not results. The implementation remains opt-in. No global collision audit,
hard injectivity guarantee, or claim that incompatible fiber layouts are solved.
