# Task

Extract the complete deduplicated union of hard failure regions from the
completed full-resolution, Classic, and best output-stride-4 evaluations.

Requirements:

- use all filtered failure episodes from the three evaluations, without the
  old 200-case quotas and without retaining model-specific labels;
- merge overlapping/nearby windows on the same reference piece;
- export GT/reference-only shortened fiber annotations, with up to 500 base
  voxels of context on either side and CP/Lasagna boundaries respected;
- audit that the full-resolution inference cache covers every exported region;
- expose the relevant high-resolution OME-Zarr channels through a readable,
  stable symlink directory for VC3D inspection;
- preserve provenance and make the extraction reproducible.
