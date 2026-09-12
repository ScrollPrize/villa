# Current snapshot tracing plan

1. Extend exact catalogue with current99/currentraw, preserving default3EMA order.
Generalize existing freeze helper to validate raw versus EMA. Freeze all before
firstGPUwork; record actualsteps rather than assuming all current files sameage.
2. Fresh outputroot; current999,current99,current order. Only Classic/latestbestpair
as baselines. Reuse approved runner, manager-exclusive full, cacheisolation/audit.
3. Print full tables aftereach before nextlaunch. Compare per-track testlosschanges
with trace scores. Verify originaltrainingroot alive and running afterfinaleval.

## Spec update

No training/numerical/protocol changes. Raw snapshots may omit snapshot_kind
(existing convention) but need test/loss_total and no EMA snapshotmetadata.
Never substitute bestweights for current. Existing shared evaluator unchanged.

## Docs updates

Preserve exact provenance/intermediate/finalreports and conclusions/commands in
docs/currenttasklog. Prior reports unchanged.

## Validation

CPU raw/EMA metadata/order/default regressiontests; three fullGPUevaluations with
100direction artifactaudits. Independent plan/finalreport review.

## Changelog

Record current snapshot comparison and testloss versus tracing findings.
