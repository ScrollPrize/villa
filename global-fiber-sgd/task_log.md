# Current task: reference-selected regional solve

Current follow-up: user selected positions5–11 (IDs14–20),requested fresh/ref
1000passes reportingevery100. Sameauditedsubset24771events,same6GPUs andLRs.
1008updates,101updatediagnostics; original initialization,no warmresume.
Output configs run_uv_window1000_{fresh,reference,suite}.json.

User additionally requested visualization ofboth1000passsolves; use existing
layout-only CPUrenderer with1000px LZW TIFFs,SVG/PDF,windingboundaries and
separatehistograms. No new referenceprojection needed for these layoutmaps.

Independent1000pass config review approved; launched75bef0a9-5c9f-43a2-9805-466f39106f91
through approvedrun_uv_window1000_suite.json. Fresh completed1000.0649passes,
L1=1.5227166,within2=80.9368%,sign1.5486%,retained95.2269%,loss2.53229.
First101update reference metrics exactly reproduce earlier100passfinal;
online loss differs because diagnostic averaging window is now100passes.

Both1000runs finishedexit0. Fresh17.558s,reference18.184s solvewalls; suite95.933s
includingmanagerwait/setup. Ref finalL1=.580748,within2=98.1752%,sign.944902%,
retained100%,loss1.593159. First101updatemetrics exact forboth previous100runs.
Finalstatesfinite,all6GPUpreflightspassed,leasefinished/releasedallresources.
Fulltwo tables and timings:uv_reference_5_11_1000_results.md.
CPUviscompletedexit0: data/uv_window1000_vis_v1/{fresh,reference}/layout.tif,
9918×1000and5770×1000,LZWverified;SVG/PDF+separatehistograms complete,noPNG.

Follow-up: user confirms positions from leftmost reference surface. Metadata
ID10 begins atcolumn1; selected IDs13–19,14–20,15–21 correspond topositions
4–10,5–11,6–12. Same Z30k–50k, sixGPUs,100sampledpasses both initializations.
No objective changes; derive budgets per extracted event count. New results
will be in uv_reference_windows_results.md, preserving prior tables.

All3extracts completed ~15s each concurrently,auto-audits passed; sources
2018/2192/2262,events22186/24771/26429. Independent config review approved;
20subset/reference tests passed. Six-run suite started via approved command,
manager task ae3cf4a9-bbf4-498a-9d28-f5ceab2511b4,exclusive6GPUs.
User asked whether undeclared surface predatesID10: checked entire actual XYZ
TIFF validity mask47,026,606vertices. Minvalidcolumn1 and0vertices outside all
declared rectangles, so leftmost reference surface does belong toID10.

Six-run suite completedexit0 in119.889s including managerwait/setup; solvewalls
2.906–3.618s. All6CUDAgrad/coordinatepreflights passed; finalarrays finite.
Finalfresh L1=2.6239/2.7427/2.9010; finalreferenceL1=.4876/.4831/.4881.
Reference within2=100%,signviolations4.6955/4.5390/4.8024%; allrunsretained100%.
Fulltables every~10passes and prior1–10baseline persisted in
uv_reference_windows_results.md. No convergence or kollesis identification claim.
Manager ae3cf4a9-bbf4-498a-9d28-f5ceab2511b4 finished/releasedallresources.

2026-10-12: User requested first ten reference windings, Z=30000–50000 base
voxels, 100-iteration fresh/reference solves with metrics and time. Plan:
uv_reference_subset_plan.md. Prior completed results remain in
uv_joint_lr_width_results.md and changelog.md.

Reference assignments span IDs10–135; use10–19 inclusive. Existing projection
has2652pieces/2651sources with at least one sampled point in this ROI (final
exact clipping may differ). Reuse earlier stages, no normal/HV/direction solve.
Independent review agrees with plan; preserve global radial gauge, clip metrics
to retained geometry and document batch/LR and boundary constraint losses.

User explicitly requested multi-GPU for speed, overriding the proposed1GPU
subset run. Use6GPUs,4096/GPU,global24576,stateLR0.0073484692,widthLR2 unchanged.

Extraction completed15.2396s:7218reference-selected sources before Z clipping,
2651retained sources,2662pieces,30048events,1continuation,4027780.258base-vx arc.
Exact output Z bounds30000/50000; reference IDs10–19. Production audit passed:
finite arrays,correct identities/owners/degrees/arcs,91191clipped metric samples.
Reference subset projection is a metric/initialization cache, not a surface
render projection (no nearest_rc/placed_rc); exact inspectable clipped curves
are separately saved. No visualization was requested this task.

Independent plan/implementation/config reviews approved.48tests passed,
2CUDA-only tests deselected; production launch includes6GPU gradient preflight.
Width diagnostics optionally filter reference intervals10–19 after offset;
initialization/prior unaffected. Started approved run_uv_subset100_suite.json
under exclusive6GPU manager task c01de29e-1070-4e4c-921d-10eb305820f6.
123updates give100.6006sampledpass equivalents,12update reporting (~9.81passes).

Completed both runs, suiteexit0, leasefinished/released. Fresh solve5.821s:
loss43.2843,referenceL1=2.3063,within2=62.1652%,retained100%,sign22.5004%.
Reference solve3.792s:loss32.3750,L1=.4885,within2=100%,retained100%,sign9.8259%.
Suite49.889s including startup/manager, excluding prior extraction. No convergence
claim: smaller dataset gives only123updates, versus3360in full100pass run.
Full checkpoint tables/timing/commands in uv_reference_subset_results.md.
Final regression50passed,2CUDA-only deselected; production6rankgradient and
coordinategradient/bitwisepreflights passed. Boundary-link hardening produces
exactly the same stored link; production audit passed, no regeneration needed.
