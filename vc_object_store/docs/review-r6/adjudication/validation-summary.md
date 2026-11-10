# Independent R6 adjudication

Baseline: `bea32d5d807731d8c3b2db17c815cde748523226`; compared dirty R5 production diff to HEAD. Audited ten candidates against implementations, public boundaries, baseline source and review contract. Source inspection only: no new code, tests, builds, installs or runtime probes.

Accepted: C-001, C-002, C-003, C-004, C-005, S-001, X-001, X-002, X-003. X-001 is recommended P1; others P2. Rejected S-002's new/unresolved P2 framing because separate-file recovery is explicitly documented and allowed by original R5 S-005.

Regressions: C-004 (specific resource ordering direction), S-001, X-001, X-002 (Python boundary). Pre-existing: C-001, C-002, C-003, C-005, X-003. No duplicate IDs collapsed, but shared symptoms/secondary callers must not inflate counts. C-004 requires an honest coarse-timestamp limitation rather than a promise to recover missing chronology.

Runtime coverage is absent for every adjudication; source-confirmed does not mean tested. Packed/sharded, hosted, mixed history and concurrent variants remain untested unless already independently verified elsewhere. This adjudication makes no coverage percentage or exhaustive correctness claim.
