# Vesuvius Challenge PGS-Recon snapshot

This directory is the PGS-Recon source validated for the Vesuvius Challenge photogrammetry workflow. It is a modified snapshot of the upstream project, not an unmodified upstream release.

## Source and authorship

- Upstream project: [EduceLab PGS-Recon](https://gitlab.com/educelab/pgs-recon)
- Upstream base revision: [`5ccaa10f4de821aaa5f98e60593fd8be69ecda3a`](https://gitlab.com/educelab/pgs-recon/-/commit/5ccaa10f4de821aaa5f98e60593fd8be69ecda3a)
- Global-scaler contribution: Giorgio Angelotti's [merge request !39](https://gitlab.com/educelab/pgs-recon/-/merge_requests/39), final head [`d2be49300dca3caa051f40dfde61c2bda304eb09`](https://gitlab.com/educelab/pgs-recon/-/commit/d2be49300dca3caa051f40dfde61c2bda304eb09)
- Upstream incorporation and review: [merge request !41](https://gitlab.com/educelab/pgs-recon/-/merge_requests/41)
- License: [AGPL-3.0-or-later](LICENSE)

The global-scaler work adds subpixel ArUco corner refinement, per-marker Umeyama scale estimation, a weighted-median summary and optional SVG histogram output. The final implementation weights a marker estimate by its number of available triangulated corners.

Follow the upstream [`CITATION.cff`](https://gitlab.com/educelab/pgs-recon/-/blob/main/CITATION.cff) authorship when citing PGS-Recon: C. Seth Parker, Kristina Gessel, Stephen Parsons, maekclena and Summer McCune. Reproducibility records for this snapshot additionally states `upstream revision 5ccaa10f4de821aaa5f98e60593fd8be69ecda3a, with global-scaler changes by Giorgio Angelotti from merge request !39 and Vesuvius Challegne-specific modifications`.

## Vesuvius Challenge-specific changes

The buildable snapshot includes:

- an Ubuntu 22.04 Docker build using the distribution-packaged ExifTool;
- `--stop-after {sfm,autoscale}`;
- `--resume-sfm PATH`;
- `--autoscale-histogram-out PATH`;
- `--autoscale-debug-dir PATH`;
- complete scaler command recording in reconstruction metadata.

When these options are absent, the legacy one-command behavior is unchanged.

## Validation

The ready-to-build snapshot was validated on 2026-08-25 from a clean Ubuntu EC2 worker. Verification included the source manifest, Docker build, command-line smoke tests, `NORMAL` global SfM without robustification, a separate histogram-producing scaling checkpoint, resume from the approved scaled scene, OpenMVS densification, reconstruction, refinement and texturing. The resulting mesh passed finite-geometry, topology, scale and visual checks on real scroll photographs.

The operational instructions are in [`../photogrammetry-docs/README_HANDOVER.md`](../photogrammetry-docs/README_HANDOVER.md).

## Newer upstream generation

A newer upstream generation exists. As checked on 2026-08-26, revision [`45598f6718a83a4095ee9ffe519429205949f29c`](https://gitlab.com/educelab/pgs-recon/-/commit/45598f6718a83a4095ee9ffe519429205949f29c) identifies itself as `v2.0.0-alpha.3-2-g45598f6` and contains a newer manifest-driven staged workflow. It is not the implementation pinned here and has not replaced this snapshot. Migration requires a separate compatibility review and end-to-end real-data validation.
