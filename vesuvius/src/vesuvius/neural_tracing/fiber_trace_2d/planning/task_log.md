# Task Log

## Findings

- Re-running the fixed-reference graph would only repeat the already converged
  conditioned solution and would not test the ordinary objective's basin.
- The joint-grid implementation initializes all factor messages uniformly.
  A MAP label vector alone therefore has no effect unless it is converted into
  initial factor messages and integer support.
- The conditioned and ordinary graphs have different components and gauges.
  Seed windings must be normalized per ordinary integer component before they
  can initialize the reference-free graph.
- Independent review found that expanded warm support is itself a finite-domain
  change. The experiment therefore includes a neutral-message solve over the
  identical expanded support before attributing any result to initialization.
- With fixed-prepass orientation, H/V cannot change. The experiment is scoped
  to Defect, winding, and component-sign attraction under fixed H/V.
- The conditioned MAP is post-projection. Conditioning ordinary factor messages
  on that MAP is an objective-neutral initialization heuristic, not a transfer
  of the conditioned graph's private BP messages.

## Deviations

- A general joint-H/V warm start would require resolving the class-sign gauge
  after the reference nodes are removed. This diagnostic intentionally requires
  the production fixed-orientation prepass and fixed half-step phase instead.

## Validation

- Release build targets `vc_fiber_trace_chunk` and
  `test_fiber_trace_winding_bp` completed.
- `test_fiber_trace_winding_bp`: 77 test cases passed.
- 1024 reference run, 1360 ordinary pieces and 69,172 input ordinary
  constraints: cold residual `3.03e-9`; conditioned residual `1.66e-9`;
  neutral-expanded residual `2.11e-9`; initialized residual `147.45` after the
  2000-message limit.
- Neutral-expanded exactly matched cold on all 1360 pieces and decoded energy
  (`759105`). The conditioned starting state had ordinary-objective energy
  `1.13393e6`; the nonconverged initialized decode reduced it to `1.08618e6`,
  differed from neutral-expanded by 42 active-to-Defect, 23 Defect-to-active,
  and 531 winding states, and remained close to the conditioned seed (6
  Defect-to-active and 30 winding changes).
- Result: `unresolved`. The conditioned state was not demonstrated to be an
  ordinary fixed point or local minimum.
