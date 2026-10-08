# Rejection accounting by relation type

User requests parallel/perpendicular rejection tables with percentages.
Existing aggregate counters cannot recover types of discarded candidates.

- Instrument only: retain initial certified relation on ScoredCandidate, including
  failures after refinement. Seed failures remain unclassified, never guess type.
- Record accepted, first-failure and deduplicated outcomes by relation. Same
  extraction/dedup choices, no measurement or database changes.
- Tests check per-type and global ledgers; approved build/test and full moving16
  raw extraction rerun, exact comparison with saved raw DB, no export overwrite.
- Report percentages within each classified population and separate unclassified
  pre-classification failures. Document initial type semantics and denominators.

Docs/spec: only diagnostics change. Update task log/status/changelog and persistent
relation_rejection_results; independent review per fiber tracing workflow.
