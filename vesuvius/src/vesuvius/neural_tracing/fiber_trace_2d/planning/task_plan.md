# Plan: Lasagna-Fallback Segment Metadata Cleanup

## Metadata Construction

1. Populate meeting error, ratio, and source only when the native trace result
   is accepted and its fused path becomes the stored segment geometry.
2. Keep failure code/detail for rejected native attempts that retain Lasagna.

## Persistence And Loading

1. Serialize native meeting diagnostics only for `accepted_native`; write null
   errors and an empty source for `lasagna_fallback` even if an in-memory caller
   supplies stale values.
2. Parse meeting diagnostics only for `accepted_native`; ignore their JSON
   values entirely for `lasagna_fallback` so earlier fallback records load.
3. Keep the configured acceptance ratio validation separate from the observed
   meeting ratio diagnostic.

## Tests

- Verify accepted native diagnostics round-trip, including an observed ratio
  above one.
- Verify fallback serialization clears stale diagnostics.
- Verify a fallback record containing legacy meeting values and an over-one
  ratio loads while retaining only its failure reason.
- Run `test_line_annotation_generated_views` and build `VC3D` with `-j32`.

## Spec Update

- Restrict persisted meeting diagnostics to accepted native segments and state
  that fallback loaders ignore discarded native meeting values.

## Docs Updates

- Clarify the accepted/fallback metadata split in the line-annotation and code
  structure documentation.

## Changelog

- Record the fallback metadata cleanup and existing-project load fix.

## Review

- Verify accepted span display and protection retain meeting diagnostics.
- Verify fallback display retains stable failure code/detail.
- Verify no native trace geometry or acceptance behavior changes.
