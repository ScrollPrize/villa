# Plan

1. Freeze subset from completed short200_best_u22000_20260916_v1: select every directed piece with reported restart_count>0, dedup exact path, preserve hash/model/GT/protocol.59pieces,62restarts across0139/1299/Paris4. No other snapshot interpretation or full50fiber replacement.
2. Existing benchmark-full runner with render=true, beam backend, same inference cache and CT bindings. Dedicated output root; demand-only manager if renderer needs new predictions. No native/render changes.
3. Add reusable review index for rendered short benchmarks, using existing reference_segments/segment_statistics. Index by original benchmark piece and source fiber; include reported restarts, failures/mm, mean and page/trace/metadata links, sorted failures/mm. Include historical origin metadata from subset manifest. Validate all expected rendered pages/cases and identifiers. No exclusion decisions.
4. Independent review, regression tests via approved runner, execute configured render, verify completion, original vs rerun outcomes/geometry, image decode/metadata. Persist docs/log/changelog and report index/paths.

## Spec update
Short benchmark opt-in render outputs a complete per-piece review index; no policy/model/cache/default change.

## Docs
Exact subset provenance/command, counts, timing and comparison; preserve paused oracle planning in archive. User will judge exclusions.

## Added user requests: vc3d exports
Copy original annotations beside renders under unchanged original filenames.
Export extracted GT and each separate trace segment under canonical vc3d names:
evalextract_TIMESTAMP_PIECE.json and evaltrace-beam/fiberlet_TIMESTAMP_SEQUENCE.json.
Timestamp is persisted per export run; sequence=pieceID for GT,1000*pieceID+segment
for traces. Include source/case/run/coordinate provenance in both derived types,
retain original geometry/source metadata, validate with canonical fiber parser.
Archive generic earlier exports in a hidden directory without deleting edits.
No artificial line across restarts. Apply to existing mini5 and59failure renders.
