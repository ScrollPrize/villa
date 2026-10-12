# Reference-selected regional solve

Current: expanded seven-fiber intact trusted solve, then constraint-vote GT
evaluation of two saved solutions. See bbox_gt_eval_plan.md; comparison artifact
identity and disabled-piece vote convention awaiting user clarification.

Current: four bbox_gt JSON fibers, independent solve; bbox_gt_plan.md.

Current: stop adaptive search now and commit the best completed result as new
benchmark baseline. Preserve all completed trials; no further solves authorized.

Current: iterative loss-weight tuning from fresh initialization, accept by final
reference L1, revisit other weights after each acceptance. Only report accepted
improvements, not 1k progress. See uv_loss_search_plan.md.

Current: fresh0.5position run withanother10xstateLR, thenwidthLR20/.2 on
stateLR selectedbyfinalreferenceL1; full1kprogress tables.

Current: increase metric position term from0.1 to0.5 of bucket term, matched
fresh/reference10k trials at current high LR; all other settings unchanged.

Current: implement bucket-only sign/distance with0.1x old positional distance;
run scratch/ref-init10k at high LR. See uv_bucket_split_plan.md.

Current: adopt measured10x state LR as future default and discuss separate
width-independent winding-index and weaker (0.1x) position losses.

Current follow-up: stop balanced baseline suite; test10x and0.1xstateLR for
fresh/reference10k each, widthLR2unchanged, same weighted objective. Report
every1k includingwidthfactor. Plan uv_class_balance_plan.md.

Current: balanced parallel/perpendicular event costs, total ratio1:10, enable
fixed sign charge. Implement/test, then matched fresh/reference10k trials.
See uv_class_balance_plan.md.

Current: userclarified10000totalpasses bothfresh/ref positions5–11,
reportevery1000passes. PreservecontinuousAdam byreproducingoriginalstarts.

Current: positions5–11 (IDs14–20) fresh/reference initialized1000passes,
report separate complete tables every100passes. Same6GPU settings and Z slab.

Current follow-up: positions4–10,5–11,6–12 (IDs13–19,14–20,15–21), both
fresh/reference initialization, sixGPUs,100passes each. Same Z30k–50k.

Current: extract reference-matched fibers in winding IDs10–19 and base
Z30000–50000, run100 sampled pass equivalents fresh and reference-initialized,
report metrics and timing. See uv_reference_subset_plan.md.

Historical tasks below are completed/superseded.

Completed current follow-up: lowLR100/1000 fresh/reference-width-initialized
joint solves, every10pass metrics, normalizedwidth comparison, TIFF/vectorvis.
HighLRfuturejobs stopped peruser. Results:uv_joint_lr_width_results.md.

Current: optimize diagnostics, then fresh and reference-initialized100iteration
joint solves, report and visualize both. uv_diagnostics_100_plan.md.

Current: uv_coordinate_plan.md. Implement and measure combined coordinate
mapping with deterministic width-gradient reduction; document thoroughly.

Next authorized: after coordinate validation, benchmark larger batches on the
improved path, then implement one distributed joint solve on GPUs0–5 (not
independent runs). Document effective global batch/LR, validate gradients and
measure exclusive-GPU speed/memory/quality. Plan follows coordinate completion.

Current: uv_lr16k_plan.md. Test larger-batch LR scaling on3exclusiveGPUs.

Current: uv_local_window_plan.md. Profile quadratic and trial local +/-N windows
without placeholders, preserve exact all-hypothesis sign penalty efficiently.

Current: uv_squared_vertical_split_plan.md. Run corrected quadratic fused solve
and report speed/reference accuracy; commit current implementation first.

Current: uv_vertical_split_plan.md. V equality at constraints, separate weak
individual V≈Z prior allowing local physical-Z reversal.

Current: uv_remove_pair_phase_plan.md. Remove only redundant pairwise wrapped
phase; keep individual phase priors and vertical alignment. Squared working choice.

Current: uv_distance_objectives_plan.md. Four fused random-minibatch joint runs
from non-reference initialization on GPUs0–3, compared to reference diagnostics.

Current: uv_pair_fusion_plan.md. Fused loss/backward, unchanged4096 minibatches
and all hypotheses; measure equivalence, profiling, memory and update speed.

Current: uv_global_accumulation_plan.md. Fresh reference initialization,10 full
gradient accumulated updates, all global variables trainable, no delay/fixing.

Current: reference_init_vis_plan.md. Render completed reference-initialized solve
with both color transfers, all TIFFs together; no optimizer modification.

Current: uv_reference_init_plan.md and reference_affine_v_plan.md. Run the full
joint solve from reference winding initialization; all global variables active
from the first update. Independently rerender the previous CCW comparison with
the entire original UV map preserved and one affine vertical alignment only.

Current: reference_physical_z_plan.md; rerender with shared physical Z/pixel mapping.

Current: uv_broken_joint_plan.md. Broken optimized from first update with fixed
per-piece rejection cost; no production rerun requested.

Current: uv_fullpass151_vis_plan.md, CCW/reference images, winding quantiles and
explanation of broken-logit training versus final rejection.

Current task: uv_full_pass_plan.md supersedes historical entries below.
Combine exhaustive full-pass traversal with joint optimization; add offset-aligned
equal-fiber reference winding L1 and within-two-windings metrics.

Current task superseding staged plans: uv_joint151_strong_plan.md. ONLY one
fresh joint winding/U/V solve, strong sign penalty from start; no extra stages.

Current task: fresh 0–150 global solve, no prior U/V checkpoint. Follow
uv_fresh151_plan.md; previous warm-expansion result does not answer this request.

Current task: uv_151_palette_plan.md. High-contrast opaque1000px/2px maps and
both resolver conventions with151hypotheses (0..150) for10sweeps.

Current correction: TIFF-only raster outputs, including inspection crops.
Convert existing detail PNGs to verified pixel-identical LZW TIFF and remove
those PNG copies; preserve full-resolution maps and projections.

Current task: implement reference_comparison_plan.md. Use smooth rectangle-derived
winding colors, original-fiber median discrete reference assignment, and place
actual fiber curves on that reference winding colored by fitted winding. No
correspondence optimization; both frozen 100-sweep solutions.

Current task: reference_surface_plan.md. Download supplied Paris4 reference
tifxyz and propose comparison/color visualization against existing fiber solves.

Current task: fiber_density100_plan.md. H/V density per inferred UV surface area,
heatmaps and profiles from100sweeps, enabled-only and including disabled fibers
at saved/recomputed most-probable locations against enabled neighbors.

Current task: uv_visualization100_plan.md. Both completed100-sweep maps,
2000px map-only LZW TIFF, separate histograms, annotated SVG and vector PDF.

Current task: hv_rejection_plan.md. H/V-specific hard disabled length, counts,
rejection probabilities and V/H comparisons. Profile U buckets10000vx.

Current task: kollesis_plan.md. Plot crowding, distance disagreement, rejection
and winding uncertainty over U and U/V; omit endpoint density. Saved20 solutions.

Current implementation: surface_plan.md. User confirms1000vx maximum H/V search,
100vx grid, joint>50%, real curves, holes, tifxyz only; first update20-sweep vis.

Current task: continue both20-sweep solutions80more sweeps (total100), leave
running while preparing surface construction; uv_continue_plan.md.

First committed current optimizer/results as c01fb1ecd. Produce complete sweep
violation/global-cost history and two vector U/V layouts with confidence opacity
and histograms below. No optimization rerun. Plan:uv_visualization_plan.md.

Follow-up: draw fitted winding boundaries on both existing layouts and previews.
