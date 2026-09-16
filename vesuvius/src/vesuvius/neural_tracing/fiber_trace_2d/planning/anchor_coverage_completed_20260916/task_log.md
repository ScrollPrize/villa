# Anchor coverage task log

2026-09-16: New anchor-only quality benchmark. Previous planning archived under short200_completed_20260916. Prior fiberlet result completed200,67failures,1485.621s; artifacts short200_fiberlet_u22000_128_scroll_20260916_v2. Canonical extraction unchanged; no graph construction in new benchmark.

Independent review: preserve owner-cell diagonal broad-phase padding, flat caps
and nearest-projection convention explicit; pooled quantiles, final partial bin,
no artificial empty/singleton gaps. Implemented shared padding helper used by
existing graph caller and new anchor-only backend. Native indexed matching,
full owner extraction/cache reuse; Python analysis only. No algorithm tuning yet.
Approved workspace build succeeded. Native tests passed23replay/85tracer/14reference;
Python/integration suite in progress. New immutable short200 anchor config and
report/inspection ranking output prepared. Whole-run GPU reservation not used.

Initial approved test suite260passed8skipped. First real attemptv1 completed77
PHerc0139 native pieces then failed in coordinator's unconditional restart-summary
correction (KeyError:sources). Added backend-aware finalizer and regression;
rerunning suite before fresh immutablev2. No scientific/extraction changes.

Completedv2 all200pieces/662.089848mm,52.461s including setup/cleanup/manager.
Approved native tests122cases;Python261passed8skipped. Native logs have no
stage=fiberlets. First4scrolls reused predictions;Paris4needed40newtiles/9chunks.
Demand GPU task b2d17067-daee-4102-85bb-94d8ab05e4dc acquired only at those misses;
runner exited0 and closed session/grant.

69141accepted anchor-to-piece matches,67862unique-position interior gaps.
Histogram cells:[0,1)67758,[1,2)101,[2,3)0,[3,4)3,>=4zero.
Median0.0701819,P950.444958,max3.73638cells (cellside32basevoxels).
No empty/single-position pieces;all117954ellipse tests had valid normals.
59/8721empty arc bins(0.676528%);lengthweighted0.365219%.45empty-bin runs,
maximum5bins. Endpointmedian0.109724,P950.594699,max4.79113cells.
Worst-gap ranking and native original GT arcs saved under outputrootv2.
Full table/commands/limitations persisted in
docs/benchmarks/short200_anchor_coverage_u22000_20260916.md.
No tuning selected; placement does not prove direction/graph connectivity.
No task scope deferred; firstv1coordinator failure preserved/documented.
