# Historical slabs and causal survival confidence

The v10 model uses live history independently at each decision. No persistent
feature state, recurrent writer, remote full-size seed crop or historical main
encoder exists. See [run commands and diagnostics](README.md).

## Observation contract

`observed_path` is a finite seed-to-head committed polyline in tracing coordinates.
It retains wrong turns and loops. Replay v6 stores the polyline once and uses
exclusive `seq_start:seq_end` indices including the current head. A prefix must
end at that head and begin at its original seed. A missing remote connection is
an error, never a synthetic straight-line bridge or annotation lookup.

There are eight slots, valid entries ordered oldest to newest. The original seed
is always first. Up to four older entries divide the seed-to-96-voxel-anchor
interval uniformly; reduce their count until every interval is at least 32 voxels.
Recent anchors are 96, 64 and 32 voxels behind the head when they are available
and at least 32 arclength voxels from the preceding selected observation. This
32-voxel minimum supersedes the initial eight-voxel older-sample proposal.
Padding does no volume reads and is gathered out before convolution.

For each observation, regress four observed positions against increasing
arclength, two tracing voxels apart. Center this six-voxel window where possible,
shifting at boundaries. Short paths and degenerate fits use the supplied seed
heading. The complete decision prefix bounds every fit, including its newest
sample. Fitting against arclength preserves traversal direction on curves.

Sample an 8x65x65 tangent-aligned slab at spacing .5: depth offsets -2,-1.5,...,1.5,
lateral offsets -16,...,16. Historical volume input is CT only. Its second channel
is the observed-path heatmap, rendered with the shared segment renderer at Gaussian
sigma one tracing voxel. Include committed vertices and exact boundaries within
plus/minus eight arclength voxels of the observation. A masked leading renderer
point prevents an artificial chord from the origin to the first vertex.
Slabs remain valid outside the current crop. Annotation membership never selects
or filters inputs. Training checks every slab footprint before any image read;
a matched pair is rejected as a unit if either member is unsafe.

## Features and attention

A shared small 3D residual encoder uses widths 8,16,32 and strides (1,2,2),
(2,2,2),(2,2,2), with one existing residual block per stage. The final 2x9x9 grid
is projected to 128 dimensions by default. Spatial position, relative translation
and rotation, log historical age, slot order and seed role are embedded inside
this encoder. Its output is only spatial tokens plus a padding mask.

Generator and scorer each own one ungated residual cross-attention module, shared
across their decoder layers. Each layer applies it after current-image attention
and before the feed-forward block. Empty history has exactly zero contribution,
including output bias. Encode once per decision, keeping gradients attached;
reuse these features for generated candidates, retries and supplied candidates.
No historical coordinates or path markings separately enter either head.

The main encoder and its local history inputs are unchanged. Dense models still
use full output-plane features; the patch token-only model still samples its
coarse patch lattice. Refinement predicts absolute replacement paths through the
same coordinate readout. Acceptance, commit limits and connection bounds retain
the existing policy. Training uses fixed masked attempt slots; inference compacts
active rows. The last attempted geometry gets 75% of geometry weight and earlier
attempts share 25%; a sole attempt gets 100%.

## Loss and replay semantics

Every independently supervised decision has weight one. Keep geometry, generated
survival and supplied-candidate coefficients 1,.5,1. Generated losses average
actual attempts and supplied losses average eligible candidates per decision.
There are no auxiliary historical decisions, streamed observation-only calls,
writer reconstruction or stale detached historical features.

The confidence scorer samples four ordered points on each proposed segment and
uses causal candidate self-attention with unrestricted observed-image/history
attention. Confidence detaches path coordinates, so its losses train the scorer,
main encoder and slab encoder without training the generator. Geometry does not
train the scorer through detached confidence feedback.

For conditional failure probabilities h[i], prefix survival is
`S[k] = product(1-h[i], i <= k)`, accumulated in FP32 log space. A first failure
at j contributes survival terms before j plus `-log(h[j])`. A correct or censored
path contributes only known survival terms. Gaps censor later hazards. Replacing,
truncating or extending a proposed suffix preserves prior scores up to kernel
rounding. Wrong-fiber labels, physical endpoints, recovery bridges and unknown
annotation boundaries retain their existing semantics.

Synthetic decisions construct the full observed prefix before truncating local
history to 128 points. Short/no-history sampling remains. Matched choice and
departure pairs vary their prefix length over available source paths and use the
same slab selector. Collected replay and resumed recovery use actual saved
committed polylines, never annotated replacement geometry.
