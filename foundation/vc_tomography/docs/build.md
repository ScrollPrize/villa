# Building and testing

Run every command from `foundation/vc_tomography` in a Villa checkout. The build links vc_reader from
`../../vc_reader` (or an installed `vc_reader` CMake package; `-DVC_READER_SOURCE=PATH` points elsewhere).

## Requirements

Required:

- CMake 3.24 or newer and a C++20 compiler
- Ceres 2.2 or newer, built with EigenSparse (`apt install libceres-dev`, `brew install ceres-solver`;
  `-DCeres_DIR=...` for a separate installation)
- FFTW3 (float and double), HDF5 (C library), zlib, libcurl
- Python 3.10 or newer

Optional:

- CUDA toolkit. Built when `nvcc` is on `PATH` (`VC_TOMOGRAPHY_CUDA=AUTO`, the default); `ON` or `OFF` forces it.
  Code is compiled for the build machine's GPU; set `-DCMAKE_CUDA_ARCHITECTURES=...` to change that.
- OpenJPEG 2.5 or newer (`apt install libopenjp2-7-dev`, `brew install openjpeg`), to read frames stored as
  JPEG 2000. Without it such scans are refused. Turn off with `-DVC_TOMOGRAPHY_OPENJPEG=OFF`.
- OpenJPH, to read float32 HTJ2K volumes. CMake downloads and checks it; `-DVC_TOMOGRAPHY_OPENJPH=OFF` omits it,
  `-DFETCHCONTENT_SOURCE_DIR_OPENJPH=PATH` uses a local copy.

## Install as a Python package

```bash
uv sync --python 3.12 --extra test
```

This builds the C++ library, the bindings and the command-line tools, and installs them into `.venv`.
`vc_tomography.devices()` lists the devices the build supports.

## Incremental builds during development

```bash
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j 16
```

The tools are written to `build/release/` beside the Python extension `_core*.so`. Reuse the same build
directory for later changes.

## Tests

```bash
.venv/bin/python -m pytest -q                                # Python tests; CUDA tests are skipped without a GPU
ctest --test-dir build/release --output-on-failure           # C++ tests
build/release/vc_tomography_calibration selftest --device cpu   # registration on synthetic data
```

The Python tests compare the CPU and CUDA implementations of every kernel with each other and with NumPy
references, and check the calibration and geometry solver on synthetic acquisitions.
