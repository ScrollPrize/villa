# Task plan
1. Reload and hash-check saved seven-case inputs/solutions; independently reconstruct physical endpoint winding differences and H/V ordering bounds.
2. Export internal violations with source IDs, event IDs, raw distances, signed winding, ordering margin, residual and seam transport. Separate reversed order from missing V-to-H next-winding steps. Aggregate concentration and overlap between large cases.
3. Recheck frozen GT reference violation counts, gauges and coverage. Assess the 20 excluded unsigned terms against the fitted solution without adding them to the objective. Distinguish contradictory measured targets from sign violations.
4. Check whether all signs can be satisfied simultaneously using difference constraints; if feasible, measure the minimum L1 cost under those bounds without replacing saved results.
5. Write an audit artifact and concise findings. Do not equate different historical SGD metrics.

## spec update
No change to the solver contract. Audit only; original solutions and input files unchanged.

## docs updates
Add sign audit report and detailed CSVs; update status and task log.

## Testing
Check reconstructed counts against saved metrics, reversed-edge invariance, primal objective and constrained-solve feasibility. Report any disagreement explicitly.

## Changelog update
Record the sign audit and its quantitative findings.

## User clarification
User means signs versus GT. Prioritize a dedicated GT-event CSV/JSON audit using the existing reference scorer, verify summary equality, classify severity, identify ordinary and GT fibers, and report global versus component gauge effects. Internal constrained alternatives are auxiliary findings, not the requested result and are not promoted.
