# Foundation

Small, mostly standalone projects that support the rest of the repository.

## Subprojects

- [comparison-website](comparison-website/) — Collects pairwise image preferences through a small FastAPI and Nginx app.
- [datasets](datasets/) **(OLD)** — Scripts for generating and registering fiber, mesh, and volumetric-label datasets.
- [kaggle-visualizer](kaggle-visualizer/) — Napari helper for browsing paired 3D TIFF volumes and binary labels.
- [obj2nml](obj2nml/) **(OLD)** — Converts OBJ meshes into WebKnossos NML skeleton annotations.
- [photogrammetry-docs](photogrammetry-docs/README_HANDOVER.md) — End-to-end photogrammetry and scroll-case handover with a pinned PGS-Recon workflow.
- [pgs-recon](pgs-recon/) — Edited fork of EduceLab's wrapper around OpenMVG/OpenMVS reconstruction pipeline with reviewed physical scaling.
- [sam2-photogrammetry](sam2-photogrammetry/) — SAM 2 fine-tuned to mask scrolls and rulers in photogrammetry images.
- [scanning](scanning/) — Small scripts for x-ray and scanning-parameter analysis.
- [scroll-unwrap-pipeline](scroll-unwrap-pipeline/) — Renders scroll unwrap videos from wrap meshes, with optional ink overlays.
- [scrollcase](scrollcase/) — Creates STL models of scroll cases for 3D printing.
- [shift-optimizer](shift-optimizer/) — Assigns shifts to people for scan sessions at the synchrotron.
- [volume-registration](volume-registration/) — Tool for finding transforms between fixed and moving Zarr volumes or meshes.
