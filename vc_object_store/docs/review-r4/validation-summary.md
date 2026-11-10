# R4 independent adjudication summary

Baseline independently checked: `2b5d1e01fd26f27b3410d1b024c3f10a73533287`. This pass reviewed each candidate against source, public contracts, reachable callers and intentional limits. No production/test files changed, installations/builds performed, or runtime reproductions claimed. Detailed evidence and qualifications are in `adjudication/<ID>.md`. Existing green tests were not treated as proof against untested failures.

| ID | Disposition | Severity | Decision |
| --- | --- | --- | --- |
| C-001 | Rejected | — | Broad closure is real, but proposed disappearing history-off endpoint conflicts with sealed-dependency/retention constraints; no supported trigger established. |
| C-002 | Confirmed | P2 | Foreign structural slices bypass retained target-ID checks. |
| C-003 | Confirmed | P2 | Loaded-tool artifact publication drops complete pin/write-generation guard; numeric convenience semantics do not cover loaded tools. |
| C-004 | Confirmed | P2 | Draft/attachment shadowing after backend object limits starves visible results; limits remain per attachment, not global. |
| C-005 | Rejected | — | Mismatched supplied selected state/view is an API-hardening opportunity, not a demonstrated supported caller failure. |
| C-006 | Rejected after implementation verification | — | Pool is an unordered_multimap, not a map; duplicate keys are accepted. Original confirmation was incorrect. |
| N-001 | Confirmed, narrowed | P2 | Failed-save restart discards corrections; healthy off/on synchronously saves and is not the trigger. |
| N-002 | Confirmed | P2 | CLI surface update preserves object pin but loses the loaded coordinate-mapping guard. |
| P-001 | Confirmed | P2 | Lasagna recomputes validity in transformed coordinates after dropping the shared mask. |
| P-002 | Confirmed | P3 | Region minimum-length documentation contradicts measured selected-geometry implementation. |
| S-001 | Rejected | — | Remote coordinate/mask-only staging is explicitly documented; arbitrary auxiliary-channel mirroring is not promised. |
| S-002 | Confirmed | P2 | Public direct copy lacks whole-operation rollback if geometry fails and caller commits; no production caller found. |
| S-003 | Rejected | — | Shared `publishPayload` explicitly refuses null bytes, preventing proposed persistent negative-cache entry. |
| S-004 | Confirmed | P3 | Oversized stamped dimensions silently narrow before binding validation. |
| S-005 | Confirmed | P3 | Accepted maximum point ID overflows an unconditional successor counter, even without repair. |
| S-006 | Confirmed, qualified | P1 | AFV report stream can truncate original input or a completed backing store; report==output requires a file-layout output. |
| S-007 | Confirmed in part | P2 | Exported server target lacks Boost dependency setup; default zstd alias allegation rejected by generated export. |

Current set after implementation verification: **17 adjudicated, 12 confirmed (1 P1, 8 P2, 3 P3), 5 rejected, 0 duplicates, 0 unresolved**. C-006's original independent confirmation was incorrect: the actual pool is an unordered_multimap and does not reject duplicate keys. Its adjudication retains the original claim with an explicit superseding correction. Rejection means the submitted evidence does not establish an actionable defect, not that the area is proven defect-free. See `fixes.md` for implementation and runtime evidence; this review originally used source traces, not reproductions.
