# Training review for the crop-only axial model

This is a recommendation for discussion, not an implemented training change.
The review was prepared while training was stopped, before restarting with the
wider 32 → 64 → 128 model. The last logged update of `axial_crop_822_run1` was 400;
it had not reached its first scheduled checkpoint at update 1,000. That short
run is insufficient to establish whether any auxiliary objective helps quality.
The kernel optimizations were tested separately without restarting training.

## What the architecture changes

Localization, confidence, reference features and contrastive queries now share
one high-resolution feature map. That map has bidirectional context and is
conditioned on the observed path and visible seed. There is no independently
trained appearance encoder whose representations must be made useful to the
geometry decoder.

This makes auxiliary embedding supervision less clearly necessary. It does not
remove the need to teach the desired decision: continue the marked fiber and
reject a plausible neighboring fiber. More visual context alone does not supply
those labels.

The field of view is still finite. When every observation distinguishing two
possible original fibers has left the crop, the model cannot infer the original
identity from those identical local inputs. Observability masks remain necessary.

## Objectives and supervision

| Component | Recommendation | Reason |
|---|---|---|
| Dense curve regression | Keep | Direct supervision of the curve we need to follow. |
| Prefix confidence on the predicted curve | Keep | Teaches whether the actual proposed continuation is safe to accept. |
| Candidate confidence on verified alternatives | Keep | Directly teaches correct-versus-neighbor decisions, including alternatives the model does not currently propose. This already uses the same classifier as prediction confidence. |
| Matched visible-seed cases | Keep a bounded fraction | Same CT/history with different valid seed observations forces use of the intended-fiber cue. These are constructed training cases, not evidence of natural rollout performance. |
| Auxiliary InfoNCE | First objective to remove in a simplification experiment | It optimizes an embedding ranking proxy. The actual decisions already have geometry/BCE supervision through the same backbone. Current embeddings also contain path/seed context, so this is no longer an isolated appearance-learning objective. |
| Neighbor-aware wrong-fiber labels | Keep | A geometrically close neighbor must not become a positive merely because it falls within the position tolerance. |
| Unknown-endpoint, crop-support, holdout and visibility masks | Keep | These establish valid targets; removing them creates incorrect or contradictory supervision. |
| Two correction iterations and their intermediate losses | Keep initially | They perform local geometric refinement, which remains relevant. Whether the stronger encoder makes them redundant needs a separate comparison. |

The proposed objective has two task-level terms: curve regression and confidence
BCE. BCE should still see both model proposals and explicitly labeled candidates.
This is partly an organization of existing supervision, since both confidence
uses already share one head. It is not a proposal to remove candidate negatives.

Removing InfoNCE is a hypothesis, not a demonstrated quality improvement. The
new architecture may still benefit from it as a regularizer. Evaluate a full-data
run without it using candidate acceptance/rejection and held-out trace behavior;
reintroduce it if those regress. A low contrastive loss by itself is not evidence
that the correct tracing decision has improved.

The current trainer requires positive `identity_weight`, and the data builder
bundles InfoNCE pairs with neighbor-label construction. Proper removal requires
separating those responsibilities and skipping unused queries/projection work;
setting a CLI weight to zero is not currently a complete implementation. Keep
bank queries required for wrong-fiber labels and candidate construction.

## Sampling and replay

The present policy has several nested distributions. At the configured matched
fraction, nominal allocations are 25% matched states, 37.5% fresh, 18.75% fixed
replay and 18.75% recent replay, before empty-pool fallbacks and support rejection.
Fresh draws are further modified by bank-following/coverage reservations and
contact, old hard-span and lateral-memory oversampling. This makes the effective
training distribution harder to understand than the number of model branches
would suggest.

Consolidate to three explicit sources: ordinary annotated following/recovery,
bank-supported ambiguous cases, and recovery states from actual tracing. Keep
broad coverage of all training fibers; do not train only on bank-covered regions.
The ambiguous source should contain both own-fiber and neighboring-fiber targets,
so appearance in the bank is never synonymous with a negative label.

The strongest candidates for removing separate quotas are the lateral-memory
resampling loop and precomputed hard spans from the earlier tracing system. Both
overlap with bank-supported ambiguous cases and current-model replay. Fold
contact locations and bank coverage into one difficulty sampler instead of
retaining multiple independent probabilities. Preserve examples of close fibers;
the suggested simplification removes overlapping selection machinery.

Keep recovery replay. The fixed bank is old, but its geometry is relabeled against
the current annotations and crop. In 512 accepted draws from the actual fixed-bank
sampler, 502 had observable identity supervision, 448 had a geometry target, and
44 of 54 confirmed departures retained a visible original-fiber reference. All
20,000 fixed-bank records lack saved seed observations; their useful identity cues
come from visible history. These checks establish supervision availability, not
that all fixed-bank states are equally valuable for the new model.

Use this bank to seed a single recovery stream, then refresh that stream with
current-model errors rather than reserving a permanent quota for old errors.
Discard states with no usable task labels before expensive CT reads. The miner
and the same shared neighbor bank remain useful under this simplified policy.

Online collection itself addresses accumulated tracing errors, not an encoder
defect. Keep it, but consider its schedule and maximum trajectory length separately
from the loss simplification. Long departed tails without any visible original
reference do not become learnable merely because we collect more of them.

## Proposed next experiment

Use the full annotation set, same shared bank, optimized axial model, existing
augmentation, optimizer, crop and resolution. Use curve regression plus shared
confidence BCE; retain matched seed/candidate examples and recovery replay.
Remove the InfoNCE auxiliary and consolidate overlapping hard-example sampling.
Keep refinement and supervision validity rules unchanged for this experiment.

Measure geometry, correct-candidate acceptance and wrong-candidate rejection
together, plus held-out trace coverage, precision, departures and wrong-fiber
distance. High rejection alone can be an always-stop solution. The stopped
400-update run is not a trained control, so this experiment cannot by itself
prove the discarded terms were harmful. A controlled comparison, if needed,
should use the full dataset rather than a small-set training exercise.

Evidence and the repeatable replay audit are in
`output/axial_speed_validation/training_replay_audit.json` and `audit_replay.py`.
