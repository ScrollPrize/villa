# Status

Task: extract and audit the full deduplicated hard-region corpus.

- [x] Read the fiber-tracing workflow instructions and identify the existing
      quota-based extractor and shared helpers.
- [x] Identify the full-resolution, Classic and best output-stride-4 inputs.
- [x] Review the implementation plan independently (delegation was unavailable;
      local consistency review recorded in the task log).
- [x] Implement quota-free union extraction and cache coverage audit.
- [x] Add the stable VC3D symlink view and manifest.
- [x] Add tests and documentation.
- [x] Run extraction, validate outputs and record results.

Final output: `/ephemeral/hendrik/fiber3d_train2/evaluations/full_failure_union_20260921_v2`.
It contains 190 deduplicated GT-only pieces totalling 573.7638 mm.  The
full-resolution cache audit requires 11,124 channel chunks; 8,421 are present
and 2,703 are missing, so 58 pieces are fully covered and 132 need inference
before they can be used as a complete corridor. Six source labels have VC3D
channel links (18 channel symlinks total).
