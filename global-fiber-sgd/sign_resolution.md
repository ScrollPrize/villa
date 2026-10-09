# Whole-fiber normal sign resolution

Current output convention: signed normals point inward (writing-side convention);
visualization arrows point outward, i.e. negative stored normals. The weak radial
prior chooses global orientation, not a per-point anatomical guarantee in folds.
Internal optimization and historical metrics retain their outward gauge; new
NPZ saves negate sign/logit and complement sign probability, with an explicit
inward marker. Unmarked historical NPZ files are outward and converted on import.
The long-fiber preview now uses640basevoxel arrows, shaft radius20 (both5x),
overwritten in place. Its index references solution_0.1_inward.npz; original
outward solution and fiber OBJ geometry are unchanged.

Measured full-volume results: [comparison table](sign_resolution_results.md).

## 3D visualization

Larger preview: `data/sign_resolution_v2/whole_scroll_1000_long_retained/`
(29.4MiB),1000longest retained fibers/25444outward arrows, no broken fibers.
Import `fibers.obj`, `signed_normals.ply`, and `umbilicus.ply` together in MeshLab.
Keep `fibers.mtl` alongside OBJ. Umbilicus is orange, radius80basevoxels, a
separately toggleable mesh. Optional `umbilicus.obj` is its thin centerline.
Reproduce via approved crop-benchmark runner with `run_1000_sign_visualization.json`.
Selection is length-biased; the earlier100fiber previews remain unchanged.

Long-fiber whole-scroll overview:
`data/sign_resolution_v2/whole_scroll_100_long_retained/`.
Select100 spatially distributed fibers from1000longest retained fibers, excluding
all model-marked broken fibers before length ranking. All candidates exceed7510
basevoxels; complete original curves, no clipping. This is length-biased spatial
coverage, not a random sample. Reproduce with the approved crop-benchmark runner
and `global-fiber-sgd/run_long_whole_scroll_sign_visualization.json`.

Ready preview: `data/sign_resolution_v2/visualization_0.1/`.
Open `fibers.obj` and `signed_normals.ply` together in MeshLab (import both as
layers, same native XYZ coordinates). Keep `fibers.mtl` beside the OBJ. Normals
are triangle meshes with shaft/cone arrowheads, not just points or viewer-specific
normal attributes. Blue arrows are retained fibers; red arrows are model-marked
broken fibers. `broken_fibers.obj` is an optional separate overlay if OBJ line
material colors are ignored; set that layer red manually. These are **sheet
normals, not fiber tangents**. Display arrow length64base voxels has no physical
normal-magnitude meaning.

OBJ compatibility correction: use one two-index `l i j` record per segment,
not one multi-index polyline record. MeshLab showed only the first segment of
the original multi-index rows. Both existing OBJ files were repaired in place;
re-copy/reload them. PLY arrows are unchanged. `obj_segment_validation.json`
records checked counts; no connections were introduced between different fibers.
The shared raw-AFV reader also dropped each fiber's first point through an
evaluation-order bug; this is corrected. Regenerated preview:45fibers,
720vertices,675segments; broken overlay61vertices/60segments. `index.json`
vertex ranges updated. Signed-normal PLY is byte-identical to the previous copy.

Preview uses penalty0.1, selecting a512base-voxel box around an endpoint of
fiber454634, the highest-perpendicular-degree broken fiber. It contains45whole
fibers (1broken),349arrows,6631mesh vertices/6282triangles. Selection uses AFV
block-AABB intersection candidates; full selected fibers are exported, **not
clipped to the box**. Normals are exact saved endpoint samples, thinned to spacing
at least256base voxels along each fiber, then multiplied by the solved sign.
No inference or GPU allocation.

Copy the entire preview directory to the viewing host. `index.json` identifies
every original source fiber, probabilities, OBJ vertex ranges, and arrow endpoint
event/arc/position/normal/PLY vertex range. Files are self-contained for viewing;
the original AFV/database are not needed on that host.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_sign_visualization.json
```

For another region, set `bounds_xyz: [[xmin,ymin,zmin],[xmax,ymax,zmax]]`, or
explicit `fiber_ids: [id,...]`; use a new output directory. Set `solution` to
another penalty's NPZ to compare. Arrow spacing/length/radius are configurable.
The exporter refuses an existing output directory or excessive fiber/point/arrow
counts rather than silently truncating selection. Raw AFV reading is reused;
no training-time repair, resampling or kink removal is applied to the curves.

This stage aligns the arbitrary baked sheet-normal gauges across original AFV
fibers. It does **not** split traces, assign winding numbers, or prove anatomical
fiber correctness. The schema2 input database is read-only; solutions are separate.

## Evidence and representation

One sign logit `u` and broken logit `v` for each of512,698whole fibers. For each
of1,320,610perpendicular events, store two dense fiber indices and signed unit-normal
dot `c` at the exact constraint endpoints. No geometry or normals are evaluated
during an optimizer iteration. Invalid endpoint events are excluded explicitly
(zero in this dataset). Parallel events are not used in this first model.

Fixed NumPy arrays: `ids`, `prior`, `i`, `j`, `dot`, `event_ids`. Tensor cache is
about41.13MiB. GPU uses float32 logits/prior/dots and int64indices. Source AFV IDs
are preserved in outputs. Cache signature includes schema, full source DB SHA256
and prior-file SHA256; prior file also records its own source DB SHA256. Replacing
the database at the same path cannot silently reuse the old priors.

## Umbilicus prior

Catalogue entry for PHercParis4/20260411134726:
https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/umbilicus/20260411134726-umbilicus-20260524235033.json

Local `data/Paris4_umbilicus.json`, SHA256
`5759e460773d23f582b9ce9dfdad1f96e0adfa2cb684101881d50c732f1e474e`.
146control points. Despite the source-annotation name endingL2, the published
JSON declares nativeL0:2.4um and32693x32693x75784, matching the AFV. Do not scale
these coordinates by4 again. Shared C++ `loadUmbilicusWithFrameCheck` verifies it;
`-Umbilicus::vector_to_umbilicus` supplies outward direction using the existing
per-Z centerline interpolation/clamping convention. Outside annotated Z extent,
the shared loader's end-center continuation applies: this is a weak prior only.

Reuse the exact prior normal-orientation sweep: regular16base-voxel samples plus
constraint endpoints, original source arc order. Accumulate signed outward dot
only on regular-grid samples using trapezoidal arc weights; endpoint insertions
have zero quadrature weight. Retain only one averaged scalar `a` and valid arc
weight per fiber. Missing normals/zero radial vectors contribute no weight;
no valid radial samples gives neutral prior0. Every regenerated signed constraint
normal must exactly match its baked counterpart (tolerance1e-12). No normal
sample arrays are serialized. Inputs are fingerprinted before/after generation.

## Objective

With `s=2sigmoid(u)-1`, `b=sigmoid(v)`:

```text
pair = sum_edges((1-bi)(1-bj) * (abs(c)-c*si*sj)/2) / sum_edges(abs(c))
prior = mean_fibers((1-a*s)/2)
broken = mean_fibers(b)
loss = pair + prior_weight*prior + broken_penalty*broken
```

The pair term is expected sign mismatch weighted by normal agreement strength.
Near-orthogonal normals provide little evidence; parallel sheet-normal axes favor
equal sign, opposite axes opposite sign. Its denominator is fixed, not recomputed
over retained events. Broken probability gates every incident event. The explicit
penalty prevents the all-broken solution from being free. Radial prior is not
gated. Repeated events/high-degree fibers carry more pair weight, while broken
penalty is once per fiber. These are **event-weighted** metrics, not unique-pair
or anatomical correctness percentages.

Adam defaults:lr0.05,prior_weight0.001,initial `u=2a` (exact0ties use0.001),
initial broken probability0.01. Fixed seed0, PyTorch deterministic algorithms.
SGD is also selectable; its learning rate needs separate tuning because losses
are normalized over the full graph. No mini-batches or stochastic data sampling.

Hard readout:sign+1 iff`u>=0`,broken iff`v>=0` (probability≥0.5). Both probabilities
and logits are saved, so other thresholds remain available. Report relaxed loss,
hard broken-fiber percentage, retained event percentage, weighted sign disagreement
on all events and on retained events. An empty retained set reports no agreement
score, not perfect agreement.43,776fibers have no perpendicular constraints and
are guided only by their radial prior; neutral-prior isolated signs are arbitrary.

## Commands

No new dependencies or installs. Existing native build and approved eval runner:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_outward_priors.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_sign_resolution.json
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_normal_sign.py
```

Set new artifact/output paths for new trials; finished outputs are not overwritten.
The managed launcher requests4GiB/4CPU ononeGPU, then starts a CUDA child through
the existing manager. The child exits before manager release. No CPU production
fallback; CPU is used for preparation and synthetic tests only.

Outputs under `data/sign_resolution_v2/`: fingerprinted`outward_priors.json`,
`tensors.npz`and metadata,`adam_sweep/summary.json`,per-penaltyhistory JSON and
`solution_<penalty>.npz`. Solutions contain IDs, sign/broken probabilities,
hard`normal_sign`/`broken`,and logits. Effective normal is solution sign × baked
normal. Input `fibers.normal_sign` is not modified automatically.

## Validation and limitations

Native142tests; five Python tests cover scalar equivalence, sign-gauge invariance
(including priors), finite-difference gradients, broken suppression/penalty,
empty graphs, hard ties, synthetic optimization and stale-source cache rejection.
No guarantee of global optimum or correct normal orientation across long gaps.
Weak global prior can fail locally near folds; measurements are not ground truth.
Low broken penalties may temporarily discard many correct fibers to avoid the
uncertainty cost of undecided sign probabilities. Report discarded coverage,
and run sufficiently long instead of interpreting low retained error alone.

The0.01trial was still re-enabling fibers at10000steps; it is not converged and
is not recommended from this first comparison.0.1is a useful initial operating
point, not a proven optimum or a calibrated probability of anatomical breakage.
