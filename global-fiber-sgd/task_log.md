# Relation-specific rejection accounting — 2026-10-08

- User requests parallel/perpendicular percentages. Prior counters aggregate
  rejected attempts; cannot reconstruct their types from retained constraints.
- Plan relation_audit_plan.md. Seed tangent/orientation failures cannot be
  classified honestly; report separately. Preserve all extraction decisions.
- Independent review approved. Native diagnostic field retains initial type
  through later failures; duplicates counted on removal, other outcomes at end.
  Added per-type/global ledger tests; fullraw rerun compares all measurement
  records against saved DB and performs no export overwrite.
- Completed359.14s runner wall:all7,368,335serialized measurements match saved
  raw DB exactly.141native tests passed; per-type/global ledgers validated.
- Parallel13,138,261attempts:29.5660%rejected,24.4026%dedup,46.0314%accepted.
  Perp12,330,351:10.3820%rejected,78.9078%dedup,10.7102%accepted.
  Unclassified7,228,591(22.1077%all):seed tangent6,367,038,orientation861,553.
- Persistent tables/commands:relation_rejection_results.md; machine-readable
  relation_rejection_statistics.json. No inferred type for seed failures.
