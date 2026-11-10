# R6 vcobj invariant review

Review target: HEAD `bea32d5d807731d8c3b2db17c815cde748523226` **plus** the uncommitted R5 implementation as present at review start. The tracked `git diff --binary --no-ext-diff` hashed by `git hash-object --stdin` is `042cc6afb3f75254b8a69df25ae0038e69af6b4f`. The untracked [review contract](../review-contract.md) and [R5 evidence](../review-r5/README.md) are review inputs. No production edits or commits are part of this review.

Discovery is split into [logical core](core-coverage.md), [physical storage and transport](storage-coverage.md), and [consumer and binding](consumer-coverage.md) lanes. Reviewers trace producer, translator, storage and consumer entry points by invariant and separately check the R5 patch for regressions. Independent Astra adjudication checks reachability, duplicates, severity and attribution for each candidate. A source-traced result is not runtime evidence.

## Result

Astra accepted nine of ten candidates: four regressions in the uncommitted R5 patch and five pre-existing issues. See the [independent decisions and severity](adjudication/validation-summary.md). The conversion report candidate was rejected as a new defect; the separate-file recovery behavior is documented.

The regressions are [metrics-only reference comparison fails](adjudication/X-001.md), [Python anchors can silently truncate point IDs](adjudication/X-002.md), [the project resource UUID helper misses legacy IDs](adjudication/S-001.md), and [mixed-history resources can reorder within one second](adjudication/C-004.md). Python now delegates anchor semantics to the native codec; its regression belongs in that shared codec's integer validation, not a second Python validator.

This pass was source-only. The lane ledgers name the entry points inspected and the cases left unverified; in particular, they do not claim an exhaustive external-caller inventory or runtime closure of the placement/history/selection matrix. No build, runtime test, production edit or commit was made. The tracked diff fingerprint remained unchanged.
