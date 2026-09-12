# Short training comparison plan

1. Add identity-bound sample-limit monitor to approved experiment controller. Bind saved launch, never mutable latest. Stop after complete evaluation AND snapshots at4700updates=601600samples. Use existing bounded TERM cleanup; test on CPU.
2. Prepare fresh five-scroll JSON601600 and combined JSON+NML499200 with same modern model, LR.01, global128, EMA.99/.999, regular run root, held-out five-scroll tests. Add optional per-dataset sampling_group and sampling_group_weight: choose scroll uniformly, then group by its explicit mass, then distance uniformly within group. Paris4 JSON/NML each half its20%; default ungrouped sampling must retain identical RNG and outputs. Reject missing/inconsistent group weights. Tests verify20% scrolls,10% subgroups and length-proportional fibers, including both Paris4 JSON directories grouped together.
3. Run sequential managed trainings; record configs/data identities and realized counts.
4. Freeze best raw and lowest-test-loss best EMA across decays for each run using existing live-evaluation helper. Reuse locked50fiber/100direction evaluator, fresh namespaces, exclusive manager ownership. Six evaluations.
5. Compare complete-reference mean segment length and per-scroll failures; resume winning run from resumable raw state, retaining EMA history. EMA inference exports cannot restore optimizer.

## Spec update

No model/loss numerics or manager protocol changes. Optional explicit sampling groups affect only requested grouped datasets; ungrouped sampling stays identical. Stop controls bind immutable launch identity and wait for completed snapshot publication, never kill a replacement. Record approximate sample boundaries. Best-test-loss selection remains unchanged.

## Docs updates

Document stop control and comparison commands/checkpoint provenance/results.

## Validation

CPU monitor tests for boundaries and identity; existing evaluation preflight; inspect checkpoint counts/EMA and complete100directions for every evaluation. Confirm manager resources released after stop.

## Changelog

Record comparison outcome and selected continuation; previous EMA task remains in committed history.
