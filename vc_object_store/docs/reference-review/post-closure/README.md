# Review after closure fixes

Frozen source: `de3afb065`, following separately committed extension decoding (`b1f242597`), shell refresh (`59910fb54`) and provenance (`75f9b7d88`) fixes. This pass is read-only for production code. Review by the existing invariant/caller contract, with separate core, consumer, root binding/facade and independent Astra lanes; no finding quotas.

Subsequent status: these three findings are implemented in [the fix tracker](fixes.md), followed by [another source review](../iteration-next/README.md). The original pre-fix evidence below is retained rather than retroactively called passing.

## Confirmed remaining issues

| Area | Severity | Evidence / bounded impact |
| --- | --- | --- |
| PC-A01: metadata JSON import and deleted identities | Medium | `Snapshots.cpp::importJson` checks only live records before lower-level insertion. Astra reproduced same-kind deleted objects and edges becoming live again with history on and off. This bypasses the shared reserved-tombstone import policy; it is not an explicit restore operation. Cross-kind collisions still reject and are not part of the finding. Referenced routing is source-traced, not runtime-tested here. |
| PC-CORE-01: saved selection position validation | Low | `Selections.cpp::decodeSelectionRecord` and sibling selection encoding/binding ingress do not consistently use the shared position validation. Astra reproduced an encoded `commit_sequence: 1.5` becoming integer 1. The saved position is silently changed; complete selected object references remain intact. No wrong-geometry or authorization bypass claim. |
| PC-S01: stale-link scope UUID normalization | Low | `linking/LinkCandidates.cpp::commitCandidatesImpl` queries edges using canonical UUID selection, but compares returned endpoints against raw scope strings. Uppercasing valid scope/touching UUIDs can silently suppress stale pending-edge removal. Astra confirmed the complete source trace; no runtime counterexample claimed. Normal detector-generated scopes are canonical, so exposure is direct native API input. |

All three are earlier omissions, not established regressions of the three newly committed fixes. They are logged for a subsequent fix pass, not silently implemented during this review. No schema redesign or new persistent identity structures are proposed. The scalar-validation finding also includes source-traced record-metadata ingress siblings; imported source counters are deliberately republished locally, so malformed source counters are not claimed to become destination publication counters.

## Coverage

See [core](core.md), [consumers](consumers.md), [root bindings/facades](root.md) and [Astra](astra.md) for per-file/symbol dispositions, adjudication, diagnostic evidence and explicit exclusions. Initial unassigned inventory entries received an additional [source-format pass](formats.md), [spatial/linking pass](spatial-linking.md) and root helper/build dispositions before this review was finalized. Source review is not proof of every numerical algorithm, asynchronous interleaving, filesystem failure or malformed-input combination. No new interactive GUI, macOS/arm64 or full-volume performance validation is claimed.

## Verification

`bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks` exits 0 on the frozen source: seven focused native tests, two repeat/umbilicus checks, five source-repair tests, ten consumer executables, 27 selector/reference/extension tests, consumer syntax checks and 66 combined fiber/surface/manifest/grower tests pass. Existing Torch/dependency warnings remain. Astra's additional disposable diagnostics are documented separately and are not misrepresented as permanent regressions for the two open findings. `git diff --check` passes.

The preceding full PR run is documented in [closure fixes](../closure-pass/fixes.md): 53 native tests, 238 store Python, 358 Spiral plus 54 subtests, and 37 Lasagna plus 26 subtests passed, with the same three baseline Spiral failures. That previous result is not a new full-suite execution in this read-only review. There is no production-code line delta in this review; only evidence records are added.
