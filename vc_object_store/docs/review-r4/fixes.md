# R4 implementation tracker

Baseline: `2b5d1e01f`. Fix only independently confirmed findings; retain rejected candidates as review evidence. Share routing, edit guards, allocation and transaction helpers rather than introducing caller-local alternatives. Preserve schema 21, binary identities and history-off semantics. No new proxy tables, per-object directories or referenced payload copies.

- [x] C-002 retained structural point IDs
- [x] C-003 loaded artifact publication guards
- [x] C-004 region limits after shadowing
- [x] C-006 withdrawn: actual cache is an unordered_multimap, so the proposed leak does not exist; regression documents duplicate-handle ownership, no production fix.
- [x] N-001 failed-save correction restart
- [x] N-002 loaded CLI coordinate mapping
- [x] P-001 transformed validity mask
- [x] P-002 region length documentation
- [x] S-002 composite direct-copy rollback
- [x] S-004 stamped dimension bounds
- [x] S-005 point-ID successor overflow
- [x] S-006 safe report publication/protected paths
- [x] S-007 installed server Boost dependency
- [x] Focused regressions, delta review and integrated validation

Use the existing fixed local runners. Temporary fixtures must clean up automatically. Record exact commands/results and remaining limits before closure. Production code growth is reported separately from tests/documentation. No full-volume or performance run is needed for this correctness pass.

## Completion and shared implementation

All **12 remaining confirmed findings** are fixed. C-006 was withdrawn after checking the actual container type, superseding both discovery and independent confirmation; its rejected status is corrected in the review index/adjudication. No unnecessary statement-cache workaround remains.

Shared changes: loaded tool writes and artifacts use complete pins, write generations and the existing mapping-check executor; artifact upload/manifest code is shared with the explicitly current numeric convenience API. Region refill shares work-stat accumulation and excludes completed candidates before backend admission without unbounded prelisting. Direct-copy rollback uses the existing operation savepoint. Lasagna propagates canonical reader validity and removes duplicate local TIFF/validity handling. Reports use shared backing-store closure protection and atomic file publication. No schema, persistent table, proxy row, per-object directory or referenced-payload duplication was added.

Implementation commits:

- `2296c7227`: shared core guards/region/allocation/copy fixes plus their native consumers and regressions.
- `8ddb85fab`: source validity propagation and stale region-length documentation.
- `876dacdde`: safe AFV report publication and installed-server package dependencies.

## Validation

Linux, existing RelWithDebInfo build, Python 3.14. No dependencies installed in this implementation pass. Commands remain fixed and argument-free:

| Command | Result |
| --- | --- |
| `bash /home/hendrik/villa3/local/scripts/run_vcobj_pr_tests` | 53 native passed, optional SSH skipped; 212 Python store passed; 122 Spiral passed, optional real-data test skipped; 14 Lasagna passed with six subtests |
| `bash /home/hendrik/villa3/local/scripts/run_vcobj_consumer_checks` | Eight consumer executables passed; umbilicus parser 1/1; Python source repairs 5/5 |
| `bash /home/hendrik/villa3/local/scripts/check_vcobj_consumers` | All 33 translation units compiled, including GUI regression includes |
| Focused core command in `core-fixes.md` | 4/4 passed, 3.96 seconds |
| Focused Python commands in `python-fixes.md` | 78 passed, one skipped, nine subtests |
| `git diff --check` plus untracked-review-document whitespace checks | Passed |

The full Python suite includes actual importer collision/replacement tests and the self-cleaning staged installation/fresh downstream server-link smoke test. This is not merely an in-tree target-name assertion. Native fixtures cover local/referenced × history-on/off loaded mapping and correction failure/retry behavior. Tests consult the same shared request validation as production.

Independent GPT-6 Astra delta review found no remaining confirmed production defect in these fixes, including focused allocation/refill review. It found one accidentally misplaced AFV test assertion block; that was restored before the successful full run. See [delta review](delta-review.md). Independent model agreement is not proof: C-006 demonstrates why implementation/type checks and regressions remain necessary.

## Size and limitations

Production delta versus `2b5d1e01f`: **207 added / 73 removed, net +134 lines**. Regressions: **302 added / two removed, net +300 lines**. Review reports and shared-contract documentation are separate. This is not a net production-line reduction: missing safety/guard/refill capabilities require code, while duplicate manifest and TIFF/validity handling were consolidated. No solver/loss, precision or approximation change is involved.

No interactive GUI, macOS, real-volume import or performance run was performed. The correction restart's actual caller is source/compile checked; runtime tests cover persistence failure/retry/reload rather than driving the full GUI event loop. Existing Spiral `PatchSamplingAtlas` nanobind shutdown-leak diagnostics remain; the suite exits successfully. Region object limits remain per attachment/draft group, not one global limit. Discarded work stays charged; this change does not alter pre-existing frame-local callback counters within a backend page. Runtime wire exclusions are optional; an older strict server can reject the refill request and must be upgraded rather than silently returning misleading results. Geometry/report sources remain external where previously external.
