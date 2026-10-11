# Reference winding comparison (archived task)

Prior acquisition details archived in reference_acquisition_task_log.md.
Plan independently reviewed; original-source median shared across solved pieces.
Fifteen tests passed, including an end-to-end export and cache-reuse regression.
Full CPU runner active with
run_reference_comparison100.json, both saved100sweep solutions,2000px TIFF/SVG.
Read reference through shared tifxyz reader, AFV through shared reader, solved
MAP via hard_geometry and raster via shared tiled LZW implementation.
Approximation: stride2 triangulated cells, four nearest cell-center candidates,
all original vertices in cell valid, long bridges rejected. Not exact bilinear.
Sampling retains original AFV vertices plus200vx arc steps.200vx matching limit.
No path optimization or reference-driven solve changes; one global palette offset.
Independent code review found no blocking current-input issue. Added sampled
surface-hole connector guard and large-UV-jump rejection. Tiny diagonal cell
crossings can be missed (not exact cell traversal); boundary cells may lose
coverage. Discrete IDs clamped to actual10..135; colors still extrapolate.
Initial full run finished CCW images then hit NumPy-int JSON serialization in
map bounds; reproduced by end-to-end test, fixed explicit float conversion.
Initial added endpoint test also caught list-to-array normalization; fixed.
Final run restarted with all fixes after15tests passed. Initial projection was
preserved but recomputed because endpoint-label semantics changed. Future reuse
requires hashed referenceXYZ/metadata/AFV and matching geometry parameters.

Final run completed exit0,705.94s. Shared8,129,117curve samples projected against
11,713,673cells. Both conventions' three TIFF/SVG views and common reference
TIFF complete; central crops inspected. Retained arc support on chosen winding:
CCW78.4301%,CW78.3939%. One global offset+38for both; median absolute local
winding discrepancies12.9936/12.9431 (rough comparison, not anatomical accuracy).
Full stats/paths and test command in reference_comparison_results.md.
No remaining requested work; no solve files changed. No commit requested.
