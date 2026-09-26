# Experimental changes

Status: design proposal; no model or training changes implemented.

Started: 2026-09-26. Baseline: `output/direct_ct_judge_run2`.

## Objective

Produce long, connected traces of the original fiber while avoiding switches to
neighboring fibers. Improving local coordinate accuracy is useful, but success
depends on preserving identity through contacts and ambiguous regions.

The next experiment should learn more directly from CT and expose enough detailed
spatial context to follow a fiber through those regions. The two agreed input
changes are:

1. Remove fiber presence from the model inputs.
2. Increase the fine CT crop size while retaining native CT sampling resolution.

A target-conditioned centerline/tube predictor is the motivating direction. Its
architecture, losses, candidate extraction, and interaction with the judge remain
to be designed; they are not settled by this document.

## Baseline

The current direct follower consumes CT and fiber presence at two scales.
Distances below are in trace-grid voxels, with eight base-grid voxels per trace
voxel. Shapes are depth by width by width; extents describe sample centers.

| Input | Shape | Sample spacing | Behind origin | Ahead of origin | Lateral extent |
| --- | --- | --- | --- | --- | --- |
| Fine | 80 × 48 × 48 | 0.5 | 16 | 23.5 | ±11.75 |
| Coarse | 88 × 32 × 32 | 2.0 | 128 | 46 | ±31 |

The fine crop already samples CT level 0 at the selected source's native spacing:
four base-grid voxels, or 0.5 trace voxels. Smaller sample spacing from that same
source would interpolate existing information rather than reveal finer detail.

Through step 12,000 of the baseline run, training loss and fixed recovery error
improved, while monitor coverage fluctuated around 49–59%. At step 12,000, the
judge removed about 12% of incorrect length while discarding about 22% of correct
length. These observations motivate examining representation and decision-making,
but do not establish that presence input or crop size caused the failures.

## Remove fiber presence from model inputs

Use CT as the image evidence for both fine and coarse inputs. Continue supplying
the seed and observed tracing history as conditioning information.

Presence is already a derived fiber signal. For a model learning centerline/tube
evidence, it creates a plausible shortcut: the model could reproduce or refine
that signal without learning enough from CT to resolve the cases where the
presence prediction is ambiguous or wrong. Actual reliance on presence has not
yet been measured; this is a design motivation, not an established diagnosis.

Removing it makes the experiment more interpretable and avoids inheriting that
input's errors. It also removes its crop reads, resampling, and input transfer.

Implementation should remove the channel and its loading path, rather than keep
a permanently zero channel. Audit the volume reader as well: it currently opens
presence and uses its shape for trace-grid bounds. A CT-only path must derive
equivalent bounds from CT metadata and explicit coordinate scales. Any use of
presence for seed selection should be documented separately from model input;
fixed evaluation seeds should stay comparable across experiments.

The eventual output should describe the continuation of the **seeded fiber**.
Generic fiber presence alone would still leave neighboring fibers indistinguishable.
History conditioning and the output supervision must express this distinction.

## Increase the fine CT crop

Increase its physical field of view by increasing the number of samples at the
existing 0.5-trace-voxel spacing. The exact dimensions and behind/ahead allocation
are still open.

Prioritize more detailed context along the tracing direction, both before and
after a contact. Include enough lateral context to show competing nearby fibers.
The goal is to expose continuity through the ambiguous region, rather than only
the appearance immediately around the current endpoint.

Keep coarse context as a complementary source of longer-range information. The
current long backward view is coarser; a larger fine crop would preserve more
detail over the region where identity must be resolved.

A larger crop must also be usable by the model. In the tube-prediction design,
consider a spatial decoder with connections to shallow, detailed encoder features.
Simply enlarging the input while discarding the relevant spatial detail later
may not provide the intended benefit.

Accessing a genuinely higher-resolution CT source is a separate possible
experiment. It is not required for this field-of-view expansion.

## Resource tradeoff

Removing presence does not fund a proportional increase in crop volume. It
halves the input channels and reduces first-convolution work, but leaves later
feature-map sizes and most model computation unchanged. In the current two
encoders, it removes only 1,296 weights.

Increasing crop volume expands intermediate activations, convolution work, CT
reads, and potentially attention cost. Measure peak training memory and
end-to-end samples per second before choosing final dimensions. Record crop
shape, physical extent, batch size, and precision with each measurement.

Choose the larger crop for the additional evidence it provides, with an explicit
resource budget; do not assume a twofold input-channel reduction permits a
twofold crop-volume increase at unchanged cost.

## Validation and remaining decisions

Keep a CT-only experiment at the original crop size as a control for the larger
CT-only crop. When introducing a tube predictor, compare those input choices
within the same architecture and training setup where feasible.

Evaluate correct connected length, identity switches, premature stops, and
performance at difficult contacts, alongside localization error and throughput.
Use fixed held-out seeds and retain the original baseline for comparison. Lower
training loss alone does not establish better tracing.

Decisions still to make:

- Fine crop dimensions, lateral width, and behind/ahead allocation.
- Training memory and throughput budget.
- How seed/history conditioning distinguishes established and tentative paths.
- Target centerline/tube representation and spatial decoder.
- Supervision for contacts and neighboring fibers; unannotated structures must
  not automatically be treated as confirmed negatives.
- How spatial predictions produce candidate paths and how those paths are selected.

## Relevant implementation

- [Current design and training](README.md)
- [Crop construction and tracing](data.py)
- [Model configuration and encoders](model.py)
- [Volume readers and coordinate scales](../volume.py)
