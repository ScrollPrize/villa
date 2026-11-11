# R8 independent adjudication

Frozen review commit: `4ae50dfb6`. All implementation evidence below was read with `git show 4ae50dfb6:<path>`, not from the worktree where fixes were already underway. This is a separate source-only adjudication under the vcobj review contract; no runtime replay, build or production changes were performed.

| Candidate | Decision | Release scope / attribution |
| --- | --- | --- |
| C-001 | Confirm, P2 | v21 project/resource ordering regression against `da50b2f4f`, and residual failure of the earlier override-stability invariant. |
| X-001 | Confirm, P2, narrowed | Fresh paired-neighbor producer no longer satisfies joint-evaluation input contract after `b0b68f351`. Missing provenance in arbitrary older solve artifacts is not independently a correctness regression; report-only remains separate. |

## C-001 — retained creation evidence disappears from the sort key

The frozen `Project.cpp::sortByAdded` uses `createdAt` followed by the oldest publication timestamp only when `reader.objectHistory(uuid, true).front().version == 1`; otherwise its timestamp component is `INT64_MAX`. `History.cpp::ObjectStore::objectHistory` deliberately resolves only the winning record's owner. `internal/RecordOverrides.cpp::createRecordOverride` copies the original creation stamp and version, not the original owner's creation publication history. A later metadata edit advances the override's version.

For same-second legacy resources A and B, choose A before B in both UUID order and original publication order. After overriding A, its history-off winning owner yields no history; a history-on winning owner starts with the later local publication, not the original creation revision. B retains its origin publication timestamp and moves ahead of A. The history-off case alone establishes a reachable failure; the history-on sibling follows the same missing-evidence branch. At `da50b2f4f`, both legacy keys were UUID ties, so this chosen fixture remained A-first before and after the edit.

This is not a request to invent chronology for genuinely unrecorded legacy history-off timestamps. The original history remains in the dependency closure, but a current-owner history API cannot recover it for this purpose. The documented fallback does not establish acceptance of reordering on a metadata-only override. Shared `projectResources` and `projectRecord` propagate this to Python/session listing and rename/copy project selection. Resolve creation evidence through a storage-owned capability or explicitly settle the compatibility policy; do not add placement/history SQL in project consumers or change `objectHistory`'s legitimate winning-owner semantics merely for sorting.

Required tests: before/after override through history-on and history-off roots, unchanged same-second sibling, project selection as well as resource order, and controls for precise stamps, original-owner legacy history, and truly unavailable legacy history. Hosted readers must delegate to the same creation-evidence semantics. No such cases were run in this adjudication.

## X-001 — active producer omitted from provenance migration

Frozen `fiber_neighbor_pair.run` writes `prepared/tensors.npz`, `prepared/gt_events.json` and `prepared/selection.json`, then supplies `prepared` as the joint arm's `baseline_root`. Its outer `inputs.json` is at a different directory and records `selected_fibers`, not the baseline source fields the new consumer accepts. Frozen `fiber_joint_gt_eval.run` unconditionally reads `baseline/inputs.json` and requires its saved source digest before assembly. With otherwise valid GT evidence, a fresh paired run therefore fails at the absent baseline manifest. This is a concrete newly broken producer/consumer boundary, not merely an unsupported historical artifact.

`fiber_split_weight_search` forwards `baseline_root` for both joint and selectable arms; those are sibling consumers to test, but forwarding a valid manifest is not itself defective. The direct workflow documentation still lists tensors/events and GT config/solution without the newly required source manifests/summary. Updating the active producer and documenting regeneration/valid input evidence closes the actionable scope. Do not “migrate” by labeling old tensor content with a digest of whichever geometry happens to be current.

### Narrowed compatibility decision

New joint solves sample store geometry against saved tensors. Requiring verifiable source evidence is appropriate under the v21 evidence contract; accepting old artifacts with no reliable evidence would reopen R7 X-002. A missing legacy baseline manifest or GT layout summary alone is therefore not a second independent correctness bug. Give an actionable regeneration path, or a supported migration that actually verifies original evidence, rather than silently restoring unchecked assembly.

The frozen `run` begins with `if settings.get('report_only'): return report(settings)`, before either new baseline or GT provenance requirement. Consequently X-001 does **not** establish a report-only legacy compatibility regression. Report paths are not new geometry-producing solves and should not acquire these requirements incidentally. Cached-result/new-solve dispatch and explicit report-only dispatch are distinct; callers needing only an old report must use the latter.

Required tests: fresh paired producer into joint consumer, both split-search joint policies, valid direct invocation, actionable rejection of unverifiable legacy solve inputs, and report-only access without new parent manifests. No tests were executed here.

## Outcome

Two independent actionable release-relevant roots remain at the frozen commit: creation-evidence continuity across overrides, and migration of an active producer to the new saved-source contract. No additional unrelated solver/numerical work is implied. In-progress fixes and their test results are outside this adjudication; they require a new frozen delta review.
