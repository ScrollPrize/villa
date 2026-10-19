# Sign disagreements against the frozen GT reference

2026-10-19. This is the comparison requested by the user, distinct from internal constraint violations. Original solutions remain unchanged and their hashes were checked before/after. The shared scorer's summary was independently rerun and matched saved metrics exactly; optional event details now expose all individual disagreements.

| Original saved solution | Perpendicular GT signs | Parallel GT signs |
|---|---:|---:|
| GT-free (2,053 fibers) | 4 / 575 (0.696%) | 9 / 4,343 (0.207%) |
| Joint with GT (2,060 fibers) | 3 / 575 (0.522%) | 0 / 4,343 |

All reported violations are exactly one integer step. All five small tests have zero integer reference error on their 41 scored relations, hence no ordering disagreements there.

## What the events mean

Joint's three violations are ordinary fibers **424632, 424646, 425073** against **GT fiber 1**. They are all V→H relations whose positive signed separation requires at least the next winding. Solved separation is zero. Thus they are **missing-next-winding errors, not actual order reversals**. Absolute normal cosines are .979546, .997407 and .914373 respectively; these are not marginal sign projections. None of these three connectors or fiber endpoint paths crosses the seam. Event IDs 44084, 44138 and 44940 are original frozen GT cross-event IDs.

GT-free has those same three errors, plus:

- One actual perpendicular reversal: ordinary fiber **291441** against **GT fiber 2**, event 31689. Measured sign is negative, solved GT-minus-test separation +1; normal cosine −.987685. GT path has four seam crossings, ordinary path and connector none.
- Nine actual parallel reversals, all ordinary fiber **423963** against **GT fiber 1** (events 43072, 43085–43091, 43093). All predict GT-minus-test +1 despite negative measured sign. These sample one fiber pair repeatedly, not nine independent fiber failures. Their measured distances are small (.0318–.1589 turns).

The joint errors therefore touch three ordinary fibers; GT-free errors touch five. All current joint errors are a subset of the GT-free errors.

## Alignment and coverage checks

The headline table aligns each disconnected component to the frozen reference by one integer median offset. With one global offset instead, GT-free has **5/575 perpendicular and 36/4343 parallel** violations. Joint stays **3/575 and 0/4343**. This is a real limitation: the objective does not establish relative gauges between disconnected components. All 13 headline GT-free violations and all three joint violations are in component 0 with offset +6, so their detailed diagnoses are unchanged by global alignment.

Scores cover 4,918 of 5,251 frozen cross events. The 333 involving reference fiber 7 are excluded because the frozen reference provides offsets only for fibers 1–6. This audit does not silently include fiber 7 or infer its label. GT signs are tested by comparing predicted fiber labels with frozen GT labels at measured cross-event endpoints; they remain dependent on the measured connector orientation and H/V convention.

Joint includes GT fibers and cross constraints in solving, so it is an assisted reconstruction, not held-out accuracy. The GT-free optimization excludes GT fibers, but dataset membership was previously GT-assisted. Neither is an independent validation dataset. Sign accuracy also does not imply exact winding accuracy: an incorrect winding can preserve ordering.

## Next concrete check

Inspect ordinary fibers 424632, 424646 and 425073 alongside the beginning of GT fiber 1 in CT. All three missing-next-step events occur at GT arcs approximately 784, 1237 and 1603 base voxels. Their strong sign projections and shared reference fiber make this a focused place to check H/V labeling, same-versus-next interpretation, and reference/trace correctness. The saved audit gives exact test/GT arclengths for visualization. Also inspect fiber 423963 against GT fiber 1 in the GT-free solution (GT arcs 4608–5082).

## Artifacts and reproduction

- [GT-free violation CSV](../../global-fiber-min-flow/data/gt_sign_audit_v1/free_gt_violations.csv)
- [Joint violation CSV](../../global-fiber-min-flow/data/gt_sign_audit_v1/joint_gt_violations.csv)
- [Complete summary and gauge sensitivity](../../global-fiber-min-flow/data/gt_sign_audit_v1/summary.json)
- Corresponding JSON files preserve complete frozen event records.

```bash
/home/hendrik/.venv_las/bin/python global-fiber-min-flow/audit_gt_signs.py
```

Run from the min-flow worktree. Inputs are read-only; only the audit directory is written. Shared reference scorer defaults and original solutions are unchanged. Auxiliary internal-sign feasibility/tie analyses were performed before the user's scope clarification; they are retained separately under `data/sign_audit_v1` and are not replacements for this GT comparison or promoted solutions.
