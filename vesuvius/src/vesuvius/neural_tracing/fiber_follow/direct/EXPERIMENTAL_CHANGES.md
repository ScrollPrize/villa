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

The next experiment combines these inputs with a history-conditioned spatial
predictor, immutable seed context, a small set of candidate continuations, bounded
delayed commitment, and training examples spanning complete contacts. This is a
combined system experiment; a sequence of full-training ablations is not a
prerequisite.

The direction is principled, but the baseline does not establish which mechanism
limits tracing. Keep the first system small enough to diagnose. Learned memory
updates, separate direction/connectivity/uncertainty heads, and recurrent training
unrolls are deferred. The representation, scoring, supervision, and commitment
contracts below must be resolved before implementation and the next full run.

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

The reviewed logs extend through step 17,000. Training loss and fixed recovery
prediction error improve, while mean monitor coverage fluctuates around 49–59%.

| Metric | Step 1,000 | Step 12,000 | Step 17,000 |
| --- | --- | --- | --- |
| Mean monitor coverage | 56.9% | 53.5% | 51.2% |
| Median followed length | 188 | 160 | 148 |
| Fixed recovery prediction error | 0.84 | 0.69 | 0.68 |

These are small fixed monitor sets: 32 rollout seeds and 32 recovery states.
Individual checkpoint differences are noisy. The recovery fixture contains no
confirmed departures, so better recovery error does not establish preservation
of identity through a switch. The observations motivate examining representation
and decision-making, but do not establish that presence input or crop size caused
the failures.

At step 17,000, the judge removes 11.5% of wrong length while discarding 14.1% of
correct length. Under the judge's paired event metric, precision falls from 80.3%
to 79.8%. None of seven departures receives an alarm after departure onset,
although two are preempted by earlier stops. These event metrics have different
label semantics from the ordinary follower rollout metrics and should not be
mixed. Extra departed-path training was enabled at step 4,000 with
`judge_departed_fraction=0.5`; simply adding more negatives is already represented
in this run, despite that option being absent from its original `config.json`.

Additional limits on the diagnosis:

- The observed-path event detector calls a sustained distance above 3.5 trace
  voxels for 3 trace voxels a departure. It does not identify a neighboring fiber.
  Distinguish confirmed identity switches, ordinary drift, and premature stops.
- Six of the seven departures at step 17,000 begin within the first 128 trace
  voxels. Loss of all original history cannot explain most initial departures,
  although it can make later detection and recovery harder.
- The judge already retains the first four sampled positions as immutable seed
  views, with spatial image tokens, alongside recent history. Seed memory has
  therefore been tried in part and is not sufficient by itself. Conditioning the
  follower's proposals on that evidence remains a separate hypothesis.
- The follower monitor uses confidence threshold 0.5. No calibration selection
  was found in the reviewed run directory. These reports characterize a fixed
  operating point, not the best coverage/error tradeoff across thresholds.

Sources: `output/direct_ct_judge_run2/log.jsonl`, the paired reports in its
`images/judge_rollouts_*_c0.5.json`, and `output/logs/direct_ct_judge_run2.log`.

## Remove fiber presence from model inputs

Use CT as the image evidence for both fine and coarse inputs. Continue supplying
the seed and observed tracing history as conditioning information.

Presence is already a derived fiber signal. For a model learning centerline/tube
evidence, it creates a plausible shortcut: the model could reproduce or refine
that signal without learning enough from CT to resolve the cases where the
presence prediction is ambiguous or wrong. Actual reliance on presence has not
yet been measured; this is a design motivation, not an established diagnosis.

Removing it tests whether direct CT learning improves tracing and removes that
input's crop reads, resampling, and transfer. It may also remove useful evidence;
neither an accuracy gain nor harmful reliance on presence has been established.

Implementation should remove the channel and its loading path, rather than keep
a permanently zero channel. Audit the volume reader as well: it currently opens
presence and uses its shape for trace-grid bounds. A CT-only path must derive
equivalent bounds from CT metadata and explicit coordinate scales. Any use of
presence for seed selection should be documented separately from model input;
fixed evaluation seeds should stay comparable across experiments.

The eventual output should describe the continuation of the **seeded fiber**.
Generic fiber presence alone would still leave neighboring fibers indistinguishable.
History conditioning and the output supervision must express this distinction.

## Choose crop, prediction, and commitment horizons together

Increase its physical field of view by increasing the number of samples at the
existing 0.5-trace-voxel spacing. The exact dimensions and behind/ahead allocation
are still open.

Prioritize more detailed context along the tracing direction, both before and
after a contact. Include enough lateral context to show competing nearby fibers.
The goal is to expose continuity through the ambiguous region. The current fine
crop sees 23.5 voxels ahead, predicts 16 forward planes at one-voxel spacing, and
normally commits four. Extra input alone does not ensure that candidates extend
far enough to resolve a contact.

Measure approach-to-separation distance, lateral spread, and required incoming
context on the training contact collection. Use these measurements and the
resource budget to choose crop extent, prediction/search horizon, and maximum
tentative-tail length together. Candidates should include stable continuation
beyond separation when the available evidence permits it. Stop when ambiguity
persists beyond the bounded lookahead; a larger crop cannot resolve a pairing
that the image evidence does not distinguish.

Keep coarse context as a complementary source of longer-range information. The
current long backward view is coarser; a larger fine crop would preserve more
detail over the region where identity must be resolved.

A larger crop must also be usable by the model. Give the spatial decoder access
to detailed encoder features through skip connections or equivalent feature
sampling. Native CT input spacing does not require every feature map and output
head to operate at native resolution. Choose output resolution for localization
needs and measure whether it preserves distinct candidate exits.

Accessing a genuinely higher-resolution CT source is a separate possible
experiment. It is not required for this field-of-view expansion.

## Predict conditioned centerline likelihoods

Replace the single coordinate proposal mechanism with a CT encoder and spatial
decoder conditioned on the seed and observed history. Start with target-centerline
heatmaps on forward planes. Preserve distinct spatial peaks instead of immediately
reducing them to a mean coordinate, which can lie between fibers. Extract coherent
candidate curves from those peaks, then refine coordinates if needed.

The current model produces one lateral coordinate pair at each of 16 forward
planes, followed by bounded local corrections. Its pooled fine-image tokens have
an effective spacing of four trace voxels. Detailed features remain available
locally, but there is no detailed spatial decoder conditioned on the target.
Condition the decoder on the target and verify that conditioning affects its
spatial predictions. Injecting prompts into every shallow stage is not itself a
requirement; the required behavior is that the same CT evidence produces different
continuations for different valid fiber prompts.

Voxel or plane likelihoods do not define a joint distribution over complete
paths. A two-branch heatmap can permit a curve that changes identity between
planes; a tube map can merge touching fibers. Geometric continuity constraints
and a history-conditioned score for each complete passage must express the
incoming-to-outgoing pairing. Local direction or neighbor affinities alone do
not guarantee this behavior. Defer separate direction, connectivity, and spatial
uncertainty heads until their added role is demonstrated. Heatmap sharpness is
not automatically calibrated confidence that a continuation is correct.

Start with centerline supervision from existing annotations. Where the target
continuation has a unique, known crossing of a forward plane, supervise its
location with an annotation-tolerance-aware target. Mask censored planes and
unresolved crossings. Decide explicitly whether each loss supervises a unique
crossing distribution or selected positive/negative spatial sites; missing
annotations must not silently become background labels. A dilated centerline is
a training representation, not an accurate physical fiber boundary.

Forward-plane prediction assumes local monotonicity along the tracing direction,
as the current representation does. Audit this assumption over the enlarged
horizon. If folds or multiple crossings are common, shorten/reorient the local
window or justify a more general 3D representation from those failures.

## Keep conditioning and state simple

Use three kinds of context in the first system:

- Immutable seed context, retaining a small spatial representation if useful.
- Recent committed observed history, with its geometry and sampled CT evidence.
- A separate tentative path for each candidate that does not overwrite committed
  history or seed context.

Committed history is an inference-time estimate and can be wrong. Train with
model-generated and perturbed histories as well as clean ones. A distant seed's
appearance or neighbor arrangement may change along a fiber; persistent context
cannot create identity evidence that the CT does not contain.

Defer a learned bank of older spatial memories and its admission/eviction rules.
Using sustained model confidence to certify memory is circular when confident
switches are the failure of interest. Revisit learned updates only after showing
that correct choices require evidence beyond seed context and recent history,
and specify training for erroneous updates, coordinate alignment, and rollback.

Explicitly train conditioning using the same CT crop with different fiber
prompts: prompting annotated fiber A should produce A's continuation, while
prompting neighboring B should produce B's. This tests whether conditioning
controls identity rather than merely supplying a heading. Annotation identities
are training labels, not model inputs. Treat this prompt-swap check as required
validation. Contact examples must retain a distinguishing prompt; do not blindly
carry over history dropout when it would erase the information needed to determine
the requested identity. Keep startup/no-history examples as a separate case.

## Resolve contacts before committing a single path

Keep a small, explicitly bounded number of geometrically distinct continuations.
Use a fixed physical lookahead window that covers contact separation where
possible, then score the complete passage against seed context and incoming
history. Commit a supported shared prefix or the supported prefix of a resolved
choice, retaining a revisable tentative tail. Specify the search width, pruning
schedule, commitment rule, and maximum unresolved extent. Stop when no continuation
is supported or ambiguity outlasts the available lookahead.

For example, when A and B touch and then separate, inspect both exits before
choosing how the incoming fiber connects. Downstream evidence in the enlarged
fine crop should be able to change an earlier tentative decision.

Candidate diversity must preserve different exits or connection patterns, not
only small perturbations of one curve. Track whether the correct continuation
survives proposal generation and pruning. Encode shared CT context once where
possible, then score candidates using their own tentative histories. Enforce
geometric continuity while allowing distinct exits; many tiny perturbations of
one path do not provide useful diversity.

Train both relative candidate ranking and absolute acceptability. Ranking alone
always selects a winner, even when every candidate is wrong. Include candidate
sets with no known valid continuation and distinguish them from sets whose
validity is unknown because annotations end or the contact is unresolved.

Initially compare candidates over the same physical horizon. Define censoring
and any later treatment of unequal lengths explicitly. Do not sum intact-prefix
probabilities from the current judge as if they were independent step scores;
that repeatedly counts overlapping evidence. Choose a complete-passage score or
a documented incremental objective consistent with how search uses it.

The current judge can stop or truncate a single proposed curve but cannot select
an alternative. Rework its useful evidence into a history-conditioned candidate
scorer that acts before commitment, with stopping as a remaining outcome. The
existing `beam/` implementation provides search/scoring concepts to reuse where
appropriate; its cone proposals and CT resolution are not requirements for this
new spatial predictor. Reuse its ranking plus binary-supervision pattern where
appropriate, with labels for continuation of the original fiber. Reusing judge
inputs or encoders does not establish that its current weights can rank branches.

Keep committed observations append-only in the first system. Branch-local
tentative observations must be discardable/rebuildable without changing the
committed cache. The current `SliceStream` rejects rewritten paths, so candidate
inspection needs separate state or explicit cache invalidation. Specify ownership
of branch histories, transported frames, and sampled features before integration.

## Train on identity decisions at contacts

Build a spatial index of annotated contacts and a reusable collection of contact
episodes. Each episode should contain incoming context, the correct continuation,
plausible neighboring continuations, and sufficient downstream extent to tell
them apart. Use both prompt directions where annotation supports them. Exclude
duplicate annotations from confirmed competing identities and mask unresolved
contacts.

The current synthetic contact generator samples random fiber pairs. Across logged
batches at steps 14,000--17,000, six extras are requested and 5.03 fall back on
average, leaving about one successful extra sequence. Departed replay supplies
12 extra sequences per logged batch in that interval. Indexing actual contacts
should supply deliberate decision examples; a general shortage of negatives is
not the established diagnosis.

Mine actual rollouts before their first failure, retaining the approach to the
contact and competing branches. Current confirmed departed states lose geometry
supervision and mainly teach rejection. Earlier decision states can teach the
continuation that prevents departure. Preserve existing replay as a source of
realistic errors, but organize these new episodes by the identity decision as
well as drift magnitude.

Prioritize additional annotation at ambiguous contacts, including neighboring
identities and the correct incoming-to-outgoing pairing. More ordinary straight
segments may be less informative than a smaller set of resolved contacts. Human
review should distinguish truly unresolved image evidence from model failures.

Keep ordinary continuations in the training mixture. Oversampling difficult
contacts changes the decision distribution and can distort stopping calibration;
record the mixture and calibrate on representative held-out traces. Split contact
episodes before augmentation, keeping both prompt directions and overlapping
views of the same contact together.

## Train complete passages without recurrent unrolling first

Use contact episodes as supervised examples, not initially as recurrent
computation graphs. From a model-generated or perturbed approach state, generate
candidate passages, label their connected continuation of the original fiber,
and train spatial prediction, relative ranking, and absolute acceptance. A centered
path on the wrong neighbor must lose to a path that preserves the target identity.
Define candidate-quality targets and unknown-label masks before implementation.

Retain DAgger/replay to expose model-generated histories and refresh candidate
sets as the policy changes. Gradients need not pass through discrete extraction
or search: spatial supervision trains proposals and candidate supervision trains
selection. Record how each head receives its learning signal.

Defer differentiable tracing unrolls and learned memory updates until a recurrent
state mechanism or measured distribution gap justifies them. Also defer the
two-anchor reconstruction task: its far endpoint is unavailable in open-ended
deployment, and it is not needed for the first contact-selection experiment.

## Combined next experiment

Introduce the following as one coherent experimental system:

1. CT-only fine/coarse inputs with an enlarged fine field of view.
2. A history-conditioned spatial decoder predicting forward-plane centerline
   likelihoods with access to detailed encoder features.
3. A bounded set of coherent candidate curves, scored over complete contacts
   with both relative ranking and absolute acceptance supervision.
4. Immutable seed context, recent committed history, and bounded delayed
   commitment with branch-local tentative tails.
5. Indexed contacts, pre-departure replay, and complete-passage training examples,
   mixed with ordinary continuations.

Defer the learned spatial memory bank and promotion policy, separate direction/
connectivity/uncertainty heads, recurrent training unrolls, and two-anchor task.

Keep the original run as a baseline. Preserve diagnostic controls such as one
candidate versus several and disabling seed context. Inference-time removal of
conditioning can shift the input distribution, so it diagnoses reliance rather
than replacing a matched training ablation. Smaller training controls can follow;
none of these changes has yet demonstrated an accuracy gain on these fibers.

## Resource tradeoff

Removing presence does not fund a proportional increase in crop volume. It
halves the input channels and reduces first-convolution work, but leaves later
feature-map sizes and most model computation unchanged. In the current two
encoders, it removes only 1,296 weights.

Increasing crop volume expands intermediate activations, convolution work, CT
reads, and potentially attention cost. Measure peak training memory and
end-to-end samples per second before choosing final dimensions. Record crop
shape, physical extent, batch size, and precision with each measurement.

Include spatial decoding, seed features, candidate scoring, and tentative state
in the resource budget. Share CT encoding across candidates where coordinates
permit it. Measure inference latency and peak memory on ambiguous contacts as
well as ordinary continuations; average training throughput alone does not
capture the cost of search. Deferred memory banks and unrolling are not part of
the first-run budget.

Choose the larger crop for the additional evidence it provides, with an explicit
resource budget; do not assume a twofold input-channel reduction permits a
twofold crop-volume increase at unchanged cost.

## Validation and remaining decisions

Before the expensive run, perform bounded checks that make the combined system
reviewable:

- Fit a small set of resolved contacts and verify same-crop prompt swapping.
- Audit annotation coverage and the unique-forward-crossing assumption over the
  chosen lookahead window; check masking at unresolved contacts and endings.
- Measure correct-continuation recall before and after candidate pruning, then
  selection accuracy when a valid candidate exists.
- Check stopping for candidate sets with no valid continuation, without treating
  unannotated candidates as confirmed failures.
- Exercise delayed commitment and branch disposal without changing committed
  observations, frames, or cache contents.
- Measure full-input training memory/throughput and contact-search latency.

An optional follow-up control is CT-only input at the original crop size versus
the larger crop within the same new architecture. This can isolate the value of
field of view without delaying the combined experiment.

Evaluate correct connected length, identity switches, premature stops, and
performance at difficult contacts, alongside localization error and throughput.
Use fixed held-out seeds and retain the original baseline for comparison. Lower
training loss alone does not establish better tracing.

Calibrate baseline and new-model thresholds on the calibration split and compare
connected correct length at comparable wrong-length rates. Longer traces must
not count as a gain merely by continuing incorrectly. Add a held-out contact set
and longer rollouts beyond the current 400-voxel monitor cap. Report contact
traversal accuracy, premature stops, confirmed identity switches, and geometric
departures separately. Use paired fiber/contact uncertainty estimates; augmented
views or two traversal directions are not independent experimental units.
Measure correct-continuation recall among candidates before and after pruning,
then selection accuracy when a correct candidate exists. This separates failures
of proposal generation from failures of scoring or memory.

Keep annotation censoring and coordinate scales explicit. Expanded crops,
episode paths, candidate paths, and seed/history evidence footprints must all
respect the held-out region; audit the existing exclusions when extending these
inputs.

Decisions still to make:

- Fine crop dimensions, lateral width, behind/ahead allocation, and output/search
  horizon, selected together from contact geometry and the resource budget.
- Training memory and throughput budget.
- Seed representation, history conditioning, and branch-local tentative state.
- Forward-plane representation, decoder resolution, target tolerance, and losses;
  handling of nonmonotone or multiply intersecting continuations.
- Supervision for contacts and neighboring fibers; unannotated structures must
  not automatically be treated as confirmed negatives.
- Candidate extraction, search width, diversity/pruning, complete-passage score,
  absolute acceptance target, and stopping/commitment rules.
- Contact indexing, episode splits, ordinary/contact training mixture, replay
  refresh, and annotation priorities.
- How to reuse judge evidence and beam supervision while isolating tentative
  observations from append-only tracing caches.

## Relevant implementation

- [Current design and training](README.md)
- [Crop construction and tracing](data.py)
- [Model configuration and encoders](model.py)
- [Geometry and confidence supervision](supervision.py)
- [Judge supervision and synthetic contacts](judge_supervision.py)
- [Judge acceptance and retraction policy](judge_policy.py)
- [Paired judge evaluation](judge_evaluation.py)
- [Shared state sampling and annotations](../data.py)
- [Shared tracing loop](../trace.py)
- [Existing learned beam search](../beam/README.md)
- [Volume readers and coordinate scales](../volume.py)

## Architectural references

- [Flood-Filling Networks](https://research.google/pubs/flood-filling-networks/):
  recurrent 3D prediction of individual segments from raw imagery motivates the
  target-conditioned spatial continuation direction, without requiring a full
  volumetric segmentation head for the first experiment.
- [SAM 2](https://arxiv.org/abs/2408.00714): streaming memory for prompted video
  segmentation motivates retaining target context. This is an architectural
  analogy, not evidence that distant fiber cross-sections retain identifying
  appearance or that a learned memory bank is needed here.

These references support design ideas; neither establishes performance for this
dataset or the proposed combination.
