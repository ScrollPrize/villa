# Task: anisotropic Gaussian fiber-presence supervision

Resume high-resolution five-scroll training from the refinement run's best
EMA-0.999 snapshot, keeping online fiber refinement enabled. Replace the
selected run's point/centerline-only presence supervision with a soft Gaussian
tube around each supervised fiber line. The tube is anisotropic in the local
fiber/sheet frame: its in-sheet width is four times its surface-normal
thickness. Apply direction supervision over the complete Gaussian support,
weighted by the Gaussian falloff. Preserve all legacy training behavior unless
the new target mode is explicitly selected. Run the replacement only on
physical GPUs 0-5; GPUs 6 and 7 are reserved for other work.
