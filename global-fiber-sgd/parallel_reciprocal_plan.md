# Disable parallel reciprocal filtering; test perpendicular projection iterations

User requests removing parallel reciprocal checks, then measuring remaining
perpendicular losses and whether more projection iterations improve them.

- Default local parallel reciprocal filtering off (central and probe), with an
  explicit legacy switch for comparisons. All other certification stays unchanged.
- Parameterize perpendicular alternating projection iterations:4(default),16,64.
  Preserve original bounded windows; do not conflate iteration count with reach.
- Count perpendicular attempts reaching reciprocal check before dedup, failures,
  and failures on fixed projection-window boundaries. Report total-attempt and
  conditional percentages, accepted parallel/perpendicular counts and runtimes.
- Native unit tests, small real-data smoke, then three full unsigned runs through
  approved runner. No database rewrite, alignment, inference or tracing.
- Docs/spec: clarify new local default, iteration parameter and residual failures.
  Persist results/task log and report denominators; no claim more events=more
  anatomically correct constraints. Independent review before full trials.
