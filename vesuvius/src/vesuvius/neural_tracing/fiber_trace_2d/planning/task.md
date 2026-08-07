# Task: provenance-driven Atlas model registration

Make Lasagna/Fiber Atlas integration resolve and automatically register a model
from portable inference provenance plus configured snapshot directories. Use
the approved minimal Atlas layout: numeric model references,
`architecture = "fiber3d/unet"`, `task = "lasagna"`, the existing
`model_training` process, a run-relative snapshot path, and its SHA-256. Store
the Villa inference commit only in inference JSON metadata.

Use Hendrik's private staging root rather than Paul's by setting the existing
manager staging config to `s3://philodemos/hendrik/lasagna`.

Repair the two existing completed inference records so they carry the resolved
Atlas identity and current known inference commit. They already contain the
correct portable/private run name; verify and preserve it.
