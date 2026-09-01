# Task log: zero-aware winding weight refinement

## Starting point

- The existing local search rejects zero and represents every coordinate as a
  power-of-two exponent relative to the positive starting tuple.
- This prevented the normal five-coordinate refinement from testing the known
  strong `0,2,2,2,1` boundary and from revisiting zero after other moves.
- Scale-first 1024 reference results before this task:
  - `0,2,2,2,1`: 6/8 exact, 1489/639 right/wrong, 69.972%.
  - `1,2,2,2,1`: 6/8 exact, 1239/878 right/wrong, 58.526%.

## Plan review

- Use a canonical tagged state (`zero` or power-of-two exponent) and immutable
  positive bases rather than raw floating-point tuple keys.
- Zero re-entry is `{base/2, base, base*2}` and cannot depend on the exponent
  from which zero was reached. An initially zero coordinate uses base `1`.
- Candidate order is dimension `0..4`; positive coordinates propose
  `{zero,/2,*2}`, exponent bounds are inclusive `[-16,16]`, and progress uses
  the deduplicated in-range count.
- Existing strict quality comparison remains the sole condition for a move;
  residual or tuple ordering only resolves selection among strict improvements.

## Implementation and validation

- Replaced exponent-only local state with five canonical tagged coordinates:
  zero or an exponent relative to an immutable positive base.
- A positive coordinate proposes `{zero,/2,*2}`. A zero coordinate proposes
  `{base/2,base,base*2}`; the 1024 run confirmed that positive `perp_0.5`
  scenarios were revisited from cache after accepting zero.
- Release `vc_fiber_trace_chunk` and `test_fiber_trace_winding_bp` built.
- Focused winding binary: 68 test cases passed.
- `git diff --check` passed.

## 1024 zero-aware refinement

Fixed phase `0.5`, scale `0.822`, hard split continuity, both hard signs at 30
degrees, linear normal confidence, cosine decision confidence, sign cost 44,
Defect cost 100, BP temperature 1.25, and starting weights `1,2,2,2,1`.

Accepted moves:

| Iteration | Weights | Exact | Right / wrong / evaluated | Accuracy |
| ---: | --- | ---: | ---: | ---: |
| start | `1,2,2,2,1` | 6/8 | 1239 / 878 / 2117 | 58.526% |
| 1 | `0,2,2,2,1` | 6/8 | 1489 / 639 / 2128 | 69.972% |
| 2 | `0,4,2,2,1` | 6/8 | 1491 / 637 / 2128 | 70.066% |

The next complete 15-neighbor round found no strict improvement. Several
`parallel_1`, `parallel_2+`, and nearby positive weights tied the final score
because those magnitude classes are inactive under the configured parallel
cutoff; quality ties do not cause moves. The selected winding solve took
6.099 seconds.

## Default promotion

- Changed the shared H/V-aware winding-class default and CLI help from
  `0,2,2,2,1` to the selected `0,4,2,2,1` tuple.
- Updated the pinned default regression and corrected documentation/spec text.
