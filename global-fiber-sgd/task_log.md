# Inward signed normals / outward visualization
- Plan inward_vis_plan.md independently reviewed. Solver retains outward internal
  gauge/history; saved NPZ outputs canonicalize inward signs/logits/probabilities,
  explicit marker and idempotent conversion. Broken fields unchanged.
- Historical outward solution preserved; generated solution_0.1_inward.npz.
  This is a global convention change, not a new solve or pointwise radial flip.
  Weak radial prior cannot guarantee anatomical orientation in every location.
- Updated existing whole_scroll_100_long_retained/signed_normals.ply in place:
  length640/radius20 (5x each),100fibers/2463arrows. OBJ SHA256 unchanged:
  d0d236b4dc530aee35da567b5a39186ab8d97156e619f24758b16e5ad4434405.
- index.json stores inward normal plus outward display_normal, updated source/hash.
  All2463binaryPLY tips validated; dimensional regression checks entire arrow5x.
- Approved runner: /home/hendrik/.venv_las/bin/python
  /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark
  --config /home/hendrik/villa2/global-fiber-sgd/run_inward_arrow_update.json
- Initial test exposed roundoff7e-15 when comparing scaled mesh coordinates;
  test uses1e-12 tolerance, implementation unchanged. No GUI verification.
