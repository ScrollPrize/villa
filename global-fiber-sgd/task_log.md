# Recommended H/V solution visualization and commit
- User accepts recommended degree_mix/penalty0.1/late entropy0.01; code committed
  before visualization:3f579c7c2(nativefeatures),6949e82fc(stage2/params/docs).
  run_hv_recommended.json freezes only recommended row for reproducible reruns.
- Plan hv_vis_plan.md independently reviewed. Reuse existing exporter, add exact
  H/V-solution/normal-solution hash+ID binding, final ternarylabels and rejection
  origin metadata. No GPU/inference/optimization rerun.
- Stratified long-fiber sample400H/400V/200broken, whole original geometry.
  Broken sample66stage1/134stage2. Intentionally not population proportions.
- Output data/hv_stage_v1/visualization_1000/ (44.419MiB),24319normal arrows.
  Separate h/v/broken_fibers.obj and *_normals.ply; blue/green/red; inward stored
  normals/outward640length20radius arrows. Orange80radiusumbilicus preserved.
- Shared geometry buffers used to subset meshes, with crossing-face checks.
  Independent artifact audit: disjoint1000IDs,exact400/400/200labels,OBJ edge lists
  contain no inter-fiber connections,allthreePLY face indices/payload counts valid.
-16Python tests pass; git diff --check clean. No local MeshLab GUI validation;
  OBJ line material support varies, separate layers allow manual colors.
- Approved export:
  /home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_hv_visualization.json
- Exact H/V solution hash:
  8a5f39a8397b6b34ac4aa521671898d067b243851a9b42797822a379fffc2618.
  Existing preview directories and full-array solution unchanged.
