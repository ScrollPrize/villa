# Lasagna-Fallback Segment Metadata Cleanup

- Persist meeting error, meeting ratio, and meeting source only for accepted
  native fiber-traced segments.
- Lasagna-fallback segments must retain their native failure code/detail but
  must not retain meeting diagnostics for geometry that was discarded.
- Loading an existing fallback record must ignore any meeting diagnostics it
  contains, including ratios greater than one.
