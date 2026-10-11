# CCW full-pass result versus reference

Output: data/reference_fullpass151_ccw_v1/ (complete; runner exit0).
fit_own: fitted layout colored by fitted winding+11.
fit_reference: same layout colored by projected reference winding.
reference_fit: fibers on selected reference winding, colored by fitted winding+11.
reference_own: reference surface colored by its own winding.
Raster:1000px high LZW TIFF,2px opaque fibers, proportional U/V scale. SVG
companions for fiber maps. No GPU, inference or optimization.

## Robust winding ranges

|Pass|CCW q10–q90|CW q10–q90|Reference q10–q90|Offset|
|---:|---:|---:|---:|---:|
|0|49–122|49–122|42–124|+12|
|1–9|Not saved|Not saved|Not recoverable on changing cohort|—|
|10|48–121|48–121|42–124|+11|

Equal-weight original source fibers, retained and reference-matched. Exact
intermediate ranges unavailable because no per-pass states were saved.

## Broken logits in this run

Each piece has151winding logits and one broken logit, jointly softmax-normalized.
Broken mass reduces the retained pair/continuation losses; a degree-weighted
rejection cost opposes this escape. Rejection cost=10 times(.5+.5degree/mean_degree).
It is a soft training alternative, not immediate deletion; pieces can return.

Inherited rejection_warmup=0.5 zeroed only the broken-logit gradients for first
five passes. Other logits still change, so broken probability was not fixed.
Passes6–10 train it too. Entropy regularization also ramps during the second half.
No subsequent stronger sign run was launched.

Final hard readout with broken_threshold=null: reject when broken is the largest
of152probabilities. This is NOT a50% threshold; diffuse winding probabilities
can lose to a broken probability below50%. Kept pieces choose their most likely
winding. Constraints touching rejected pieces are omitted from hard consistency
metrics; reference accuracy/visualizations likewise exclude rejected pieces.
Data are not deleted: full logits remain in solution.npz.

CCW final:27950/489663pieces rejected(5.7080%),13.6649%oflength rejected.
Mean broken probability=5.7378%. Reference L1=11.5170 after+11globaloffset.
