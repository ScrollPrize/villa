# Plan: limit Fiberlet crop trace attempts

## Implementation

1. Add an unlimited-by-default maximum-attempt count to the shared crop trace
   configuration.
2. After skipping each inactive/covered seed, stop before changing or tracing
   the next active seed when either the attempt limit or accepted-fiber limit
   is reached. Covered anchors do not consume attempts; no-edge failures and
   accepted one- or two-sided traces do.
3. Expose the setting as `--max-attempts N`, with zero meaning unlimited.
4. Preserve descending prediction-presence ordering and deterministic storage
   key tie breaking.

## Tests

Add focused traversal regressions for strongest-first and storage-key tie
ordering, covered-seed skipping, no-edge attempts, zero/unlimited and exact
off-by-one limits, and interaction with the accepted-fiber limit. Explicitly
reject negative count arguments. Build the CLI and run the crop trace tests.

## Spec update

Document independent attempt and accepted-fiber limits.

## Documentation updates

Add `--max-attempts` to the crop CLI controls and explain strongest-first seed
ordering.

## Changelog

This is a small CLI/traversal control and needs no durable changelog entry.
