# vc_tomography

vc_tomography reconstructs volumes from the raw projections of synchrotron X-ray scans. It reads the detector's
frames and the acquisition record directly, measures the scan's geometry and intensity from the frames, and
writes slices or OME-Zarr volumes.

It is a C++20 library with Python bindings (nanobind). Every image operation has a CPU implementation
(OpenMP, FFTW) and a CUDA implementation (cuFFT) with the same results. Least-squares fitting uses Ceres.

## Scans it can reconstruct

| Scan | Source format | Reconstructed as |
|---|---|---|
| ESRF BM18 helical pair (`HA` + `TA`): two helical scans of one sample, the rotation stage moved sideways between them to cover a field wider than the detector | BLISS master (`.h5`) with plain or JPEG 2000 frames | one merged volume |
| ESRF BM18 multi-column scan (`HA`, `TA`, `TA2`, `TA3`, ...): more scans moved sideways in the same way | BLISS master | one merged volume |
| Diamond I12 grid: circular (non-helical) scans at several positions across the beam and several heights | NeXus NXtomo (`.nxs`, written by GDA) | one merged volume |
| ESRF BM18 mosaic: helical scans of one sample at different positions on the rotation stage (tiles) | BLISS master | each tile on its own (no merged volume yet) |

Frames can also be read at 1/2, 1/4 or 1/8 resolution, either from the JPEG 2000 files directly or from a
projection store (a multiscale Zarr copy of the frames).

## Apps

| Command | What it does |
|---|---|
| `vc_tomography_calibration` | Calibrates an acquisition: `geometry` measures the detector distortion and the position of every scan, `intensity` reduces darks and flats and builds the weight maps, `check-record` checks the acquisition record against itself and the frames. |
| `vc_tomography_reconstruct` | Reconstructs single slices or a range of slices to TIFF or OME-Zarr. |
| `vc_tomography_composite` | Renders max and mean composites along a tifxyz surface, from a reconstructed volume or directly from the projections. |
| `vc_tomography_download_projections` | Downloads the frames for a slab of a scan, optionally at reduced resolution, optionally straight into a projection store. |
| `vc_tomography_projections_zarr` | Combines downloaded reduced-resolution copies into one multiscale projection store. |

`networks/denoising/` holds a Noise2Inverse denoiser for reconstructed volumes (needs the `denoise` extra; see
[Denoising](docs/denoising.md)). `evaluation/` holds scripts that compare reconstruction methods and
measure resolution.

## Documentation

- [Building and testing](docs/build.md)
- [Reconstructing a scan](docs/reconstruction.md)
- [Calibration and alignment](docs/calibration.md)
- [Denoising](docs/denoising.md)
