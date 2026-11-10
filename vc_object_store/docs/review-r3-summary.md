# R3 coherent-store review

Baseline: `cf8406afc`. The review found **16 findings**. Three agents covered core, native consumers and Python/bindings; the primary reviewer covered storage/transport/import/linking and deduplicated findings. No finding quota was imposed. The review reports retain baseline evidence; implementation and validation are tracked below.

Implementation follow-up: user authorized dependency restoration and all fixes. Native Ubuntu test packages restored; the previously blocked probe reproduced T3-2 (`direct after=1`, `cached after=0`). All 16 fixes are implemented with permanent regressions. Reports: [core fixes](review-r3-core-fixes.md), [native fixes](review-r3-native-fixes.md), [Python fixes](review-r3-python-fixes.md), [storage fixes](review-r3-storage-fixes.md).

## Findings

| ID | Priority | Problem | Evidence |
| --- | --- | --- | --- |
| C3-1 | P1 | First inherited deletion is absent from retained history/change feed | Reproduced |
| N1 | P1 | Surface/correction editors invalidate each other's baseline; failed correction saves can be followed by discarding edits | Code trace |
| N2 | P1 | Stale fiber saves rebuild current relationship membership and can remove unseen branches | Code trace |
| C3-2 | P2 | Keeping an unchanged inherited relationship creates an override/version change | Reproduced |
| C3-3 | P2 | Unrelated registration changes length filtering/order for identical selected geometry | Reproduced |
| C3-4 | P2 | Composite native helpers leave partial writes when errors are caught before outer commit | Surface reproduced; umbilicus/catalog-transform siblings traced |
| N3 | P2 | Historical plane overlays use current geometry, potentially with old normals | Code trace |
| N4 | P2 | Point/correction editors reuse retained IDs after restore/delete and reopen | Code trace |
| N5 | P2 | Coordinate-binding dialogs guard a fresh head instead of displayed state | Code trace |
| N6 | P2 | Historical surfaces display current or stale Lasagna attached results | Code trace |
| PY-R3-1 | P2 | Old patch matches receive a current source digest and pass freshness checks | Reproduced |
| PY-R3-2 | P2 | H/V verification ignores the constraints' recorded source snapshot | Reproduced |
| T3-1 | P2 | Packed compaction fails to relocate relative external locators | Reproduced |
| T3-2 | P2 | Hosted disk cache preserves stale graph membership across refresh | Reproduced after dependency restoration |
| T3-3 | P2 | Unresolved dataset paths wrongly split or merge track groups | Code trace |
| T3-4 | P2 | Concurrent transform-choice creation can make the graph unreadable | Code trace |

Eight findings had focused pre-fix runtime reproductions; eight were code-traced. Reproducing one C3-4 member did not mean all siblings were executed; permanent regression coverage now includes surface, umbilicus and catalog-transform failure rollback. Detailed baseline locations, triggers and regression requirements: [core](review-r3-core.md), [native consumers](review-r3-native-consumers.md), [Python/bindings](review-r3-python.md), [storage/transport](review-r3-storage.md).

## Why these remained

N1 contains a concrete last-fix integration regression: the newly guarded correction snapshot is not advanced when that same editor saves its surface. Its reverse surface-pin problem predates the fix. The other findings are missed siblings/older paths, not demonstrated new regressions. Earlier fixes often covered one helper/caller or downstream validator but not another variant or evidence producer: retain versus keep-recorded; pack versus compact; index consumption versus match production; fiber/Spiral point allocation versus ordinary point collections.

Fix these families through shared APIs and caller matrices, not 16 independent workarounds: loaded editor/graph baselines, selected evidence/geometry identity, operation-wide transaction boundaries, publication/no-op semantics, and physical source relocation/identity.

## Coverage and limitations

Native review dispositioned all 150 discovered native paths plus indirect wrappers. Python review inventoried 60 consumers (55 runtime-boundary inspections, five documented exclusions) and all 13 binding/helper source files. Core and storage reports inventory shared SQL/history/geometry/extensions/query/publication and import/copy/VFS/session/transport/linking paths. These overlapping inventories are not a unique-file count.

Unchanged code was included; inspection did not stop at an arbitrary number of findings. Coverage means relevant store control flow and callers, not every unrelated numerical/UI body or every possible input/interleaving. No full GUI, macOS, full-volume or performance run. Previous suite totals are not proof that these new scenarios pass.

The non-finite sparse-byte candidate mentioned in commentary was excluded: the generic byte API explicitly permits preserving those bits; typed geometry has a separate contract. Hypothetical identity failures without reachable triggers were excluded too. The original read-only review changed neither production code nor dependencies; that restriction ended with the user's explicit implementation/dependency-restoration request.

## Follow-up checklist

- [x] C3-1 inherited deletion history/feed
- [x] C3-2 keep-recorded no-op
- [x] C3-3 selected-geometry lengths
- [x] C3-4 atomic composite writes, including catalog transforms
- [x] N1 coordinated editor baselines and save-failure propagation
- [x] N2 loaded graph membership and peer references
- [x] N3 historical plane overlays
- [x] N4 retained collection point allocation
- [x] N5 loaded coordinate-binding guards
- [x] N6 historical attached results
- [x] PY-R3-1 match evidence source guard
- [x] PY-R3-2 H/V constraint source guard
- [x] T3-1 compact relocation
- [x] T3-2 graph-cache freshness
- [x] T3-3 resolved dataset grouping
- [x] T3-4 transform-choice membership guard

## Validation

Linux, RelWithDebInfo, Python 3.14; self-cleaning synthetic fixtures, no full-volume benchmark. Commands are stable argument-free local runners; tracked tests contain the regression implementations.

Authorized environment repair restored Ubuntu development packages: `libsqlite3-dev libssl-dev libcurl4-openssl-dev nlohmann-json3-dev libboost-dev libopencv-dev libtiff-dev libblosc-dev liblz4-dev libzstd-dev qt6-base-dev libqt6opengl6-dev libxtensor-dev libxsimd-dev libspdlog-dev libeigen3-dev`. No bootstrap script, Python dependency change or repository-wide setup was run.

| Command | Result |
| --- | --- |
| `bash /home/hendrik/villa3/local/scripts/run_vcobj_pr_tests` | 53 native passed, optional SSH skipped; 202 Python store passed; 122 Spiral passed, optional real-data test skipped; 13 Lasagna passed |
| `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks` | Six focused native tests passed (4.56 s); the subsequent dataset-filter regression also passed in the full suite |
| Focused Python commands in the Python fix report | 71 passed, including producer/index/consumer and H/V source guards |
| `bash /home/hendrik/villa3/local/scripts/run_vcobj_consumer_checks` | Eight native consumer executables passed, including the local/referenced × history-on/off correction fixtures |
| `bash /home/hendrik/villa3/local/scripts/check_vcobj_consumers` | 33 translation units passed, including the controller regression includes under `VC_TESTING` |
| `git diff --check` | Passed |

The existing Spiral `PatchSamplingAtlas` nanobind shutdown leak warning remains; the test process exits successfully. No macOS, full GUI runtime or full-volume/performance result is claimed. Controller regression includes are compiled, not executed through a full VC3D GUI binary.

Delta review also caught and corrected two integration hazards before closure: plane scrolling must reuse resident geometry and the existing background/bounded query path rather than rebuilding full grids each time; selecting another version with the same UUID must refresh editor state as well as attached results. Correction-save failures veto selection changes before replacement. These are N1/N3/N6 integration closure, not a new unbounded review round.

A clean correction editor does not request a write: identical desired and loaded UI collections leave the optimistic baseline untouched, even after external changes, so navigation is not trapped. Actual changed UI data still rejects stale saves. The final consumer rerun passed this regression in all four history/reference combinations. A second reviewer confirmed closure of the two GUI integration findings; core/storage delta review found no additional concrete defect.

## Scope and commits

- `c33ad02e3`: shared core/query/transaction and storage/transport fixes.
- `0f93ab281`: Python evidence producer/consumer guards.
- `44e467ec7`: shared attached-document API and VC3D editor/overlay fixes.

Production delta versus `cf8406afc`: **526 added / 296 removed, net +230 lines**. Tests: **462 added / 20 removed, net +442 lines**. Review/fix reports and shared-contract documentation are counted separately. This pass does not achieve a net production-line reduction: it adds missing editor coordination, checked graph/cache freshness and evidence guards while removing the duplicate length-query path and sharing resident query and relocation handling. Schema 21, binary identifiers and history-off retention remain unchanged; no persistent tables, proxy rows, per-object directories or referenced geometry copies were added.

Intentional limits: measured-length queries may read geometry where the old inconsistent path trusted ingestion summaries; equal-length ties now consistently use UUIDs. Frozen Python evidence missing its original source version must be regenerated. External commits conservatively invalidate edited fiber/correction graph baselines until reload. Truly unavailable history-off evidence remains an error, not an excuse to substitute current data.
