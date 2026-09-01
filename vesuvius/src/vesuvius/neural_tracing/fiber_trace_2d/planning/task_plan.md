# Plan: zero-aware winding weight refinement

## Search semantics

1. Represent each coordinate canonically as either `zero` or an integer
   exponent relative to an immutable positive base. Cache this tagged
   five-coordinate state rather than floating-point arithmetic results.
2. For each nonzero coordinate, propose zero, `/2`, and `*2`. For a zero
   coordinate, propose `{base/2, base, base*2}` so later iterations can
   re-enable it independently of search history. The base is the supplied
   positive starting value, or `1` when that starting coordinate is zero.
3. Generate dimensions `0..4` in fixed order, deduplicate candidates, skip the
   current tuple, clip positive exponents to `[-16,16]`, and preserve the
   existing strict quality ranking and deterministic tie behavior. The
   progress denominator is the actual deduplicated candidate count.
4. Print every evaluated tuple and accepted move using the existing progress
   and ranking output.

## Tests

- Build the Release CLI and focused winding test binary.
- Run the focused winding tests and `git diff --check`.
- Smoke-test from the benchmark output that the search evaluates zero for all
  five coordinates and evaluates positive re-entry candidates after an
  accepted zero move.
- Run the established 1024 benchmark to a local optimum.

## Spec update

Document tagged cache identity, immutable bases, reversible zero-neighbor
semantics, deterministic order/range clipping, and the resulting maximum of 15
one-coordinate candidates per local iteration.

## Docs update

Update the CLI search description and record the measured 1024 result.

## Changelog

Record support for reversible zero-valued local-search coordinates.
