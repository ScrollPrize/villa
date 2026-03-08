# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Volume Cartographer (VC3D) is a C++23 toolkit for virtually unwrapping volumetric datasets, specialized for recovering text from CT scans of ancient Herculaneum papyri. It includes a Qt6 GUI application and 35+ command-line tools.

## Build Commands

```bash
# Configure (from repo root)
cmake -B cmake-build-minsizerel -DCMAKE_BUILD_TYPE=MinSizeRel

# Build the main GUI app
cmake --build cmake-build-minsizerel --target VC3D -j12

# Build everything
cmake --build cmake-build-minsizerel -j12

# Build and run tests (must enable VC_BUILD_TESTS)
cmake -B cmake-build-minsizerel -DCMAKE_BUILD_TYPE=MinSizeRel -DVC_BUILD_TESTS=ON
cmake --build cmake-build-minsizerel --target all -j12
cd cmake-build-minsizerel && ctest

# Run a single test (after building with tests enabled)
cd cmake-build-minsizerel && ctest -R <test_name>

# Format code before committing
git add . && git clang-format && git add .
```

Build outputs go to `cmake-build-minsizerel/bin/` (executables) and `cmake-build-minsizerel/lib/` (libraries).

## Key Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `VC_BUILD_TESTS` | OFF | Build unit tests |
| `VC_USE_OPENMP` | ON | OpenMP parallelization |
| `VC_USE_PCH` | OFF | Precompiled headers |
| `VC_ENABLE_ASAN` | OFF | AddressSanitizer |
| `VC_ENABLE_UBSAN` | OFF | UndefinedBehaviorSanitizer |

## Architecture

### Directory Layout

- **`core/`** - Main C++ library (`vc_core`), the heart of the project
  - `include/vc/core/types/` - Data types: `Volume`, `Segmentation`, `VolumePkg`, `ChunkedTensor`
  - `include/vc/core/util/` - Algorithms: `Slicing`, `QuadSurface`, `Geometry`, `Zarr`, rendering
  - `include/vc/core/cache/` - Tiered chunk caching system (`TieredChunkCache`, `DiskStore`, HTTP fetching)
  - `src/tracer/` - Neural tracer for surface growing
  - `test/` - Unit tests (custom lightweight GTest-compatible framework in `test.hpp`)
- **`apps/VC3D/`** - Main Qt6 GUI application. `CWindow` is the central class. Split into controllers: `SegmentationCommandHandler`, `SurfacePanelController`, `MenuActionController`, etc.
- **`apps/src/`** - CLI tools (prefixed `vc_`): rendering, segmentation, format conversion, geometry processing
- **`apps/diffusion/`** - Winding number computation
- **`utils/`** - Header-only C++23 utility library: `zarr`, `lru_cache`, `thread_pool`, `http_fetch`, hash utilities
- **`libs/`** - Vendored libraries: `OpenABF` (angle-based flattening), `cc3d`, `dijkstra3d`, `edt`, `flatboi`, `libigl_changes`
- **`cmake/`** - Build modules: `VCFindDependencies.cmake` (dependency config), `VCCompilerFlags.cmake`, `VCSanitizers.cmake`

### Key Dependencies

Qt6 (Widgets/Gui/Core/Network/Concurrent), Eigen3, OpenCV, Ceres (optimization), nlohmann/json, xtensor/xtl/xsimd, c-blosc (compression), CURL, libtiff, Boost (program_options), OpenMP.

### Data Flow

Volumes are stored in OME-Zarr format. The caching layer (`vc::cache`) supports local filesystem and HTTP chunk sources with tiered memory+disk caching. Segmentations are quad mesh surfaces stored as TIFF XYZ point clouds. The tracer grows segmentations along surface predictions.

## Code Style

- **Formatting**: `clang-format` (Google-based, 140 col limit, Linux braces, 4-space indent). Run `git clang-format` before commits.
- **Header guards**: `#pragma once`
- **Classes/Methods**: `CamelCase` (e.g., `QuadSurface`, `LoadMetadata()`)
- **Member variables**: `camelBack_` with trailing underscore
- **Namespaces**: `lower_case` (e.g., `vc::cache`). Many core types are in global namespace.
- **Pointers**: Left-aligned (`int* ptr`)
- **Include order**: system → C → VTK/ITK → OpenCV → local (managed by clang-format priorities)
